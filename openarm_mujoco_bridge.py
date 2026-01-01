#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import mujoco
import glfw
import numpy as np
import sys

# ---------------- 配置区域 ----------------
MODEL_XML_PATH = "src/openarm_mujoco/v1/openarm_bimanual.xml"

# PD 参数分层配置
KP_BASE = 1500.0      # 根部关节 (Joint 1-4)
KD_BASE = 50.0

KP_WRIST = 200.0      # 腕部关节 (Joint 5-7)
KD_WRIST = 10.0

KP_GRIPPER = 100.0    # 夹爪
KD_GRIPPER = 2.0
# ----------------------------------------

class OpenArmDynamicsBridge(Node):
    def __init__(self):
        super().__init__('openarm_dynamics_bridge')
        
        # 1. 加载模型
        try:
            self.model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            self.get_logger().error(f"MuJoCo Load Error: {e}")
            sys.exit(1)

        # 2. 构建映射表 - **关键修复**: 分别记录 qpos 和 qvel 地址
        self.joint_qpos_map = {}  # 名字 -> qpos 地址
        self.joint_qvel_map = {}  # 名字 -> qvel 地址
        self.actuator_map = {}
        
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.joint_qpos_map[name] = self.model.jnt_qposadr[i]
                self.joint_qvel_map[name] = self.model.jnt_dofadr[i]  # 使用 dofadr

        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name:
                self.actuator_map[joint_name] = i

        # 3. 初始化目标位置为当前位置
        mujoco.mj_forward(self.model, self.data)
        self.target_positions = {}
        for joint_name in self.actuator_map.keys():
            if joint_name in self.joint_qpos_map:
                q_addr = self.joint_qpos_map[joint_name]
                self.target_positions[joint_name] = self.data.qpos[q_addr]
        
        self.get_logger().info(f"初始化 {len(self.target_positions)} 个关节目标位置")

        # 4. ROS 订阅
        self.create_subscription(JointState, 'joint_states', self.ros_cb, 10)

        # 5. 图形界面
        if not glfw.init(): sys.exit(1)
        self.window = glfw.create_window(1200, 900, "OpenArm Dynamics", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.mjr_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        
        self.cam.lookat = np.array([0, 0, 0.5])
        self.cam.distance = 2.0
        self.cam.azimuth = 90
        self.cam.elevation = -30

        # 6. 物理循环
        self.create_timer(0.01, self.physics_loop)

    def ros_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name in self.joint_qpos_map:  # 安全检查
                self.target_positions[name] = pos

    def get_pd_gains(self, joint_name):
        """根据关节名称返回合适的 PD 参数"""
        if "finger" in joint_name.lower():
            return KP_GRIPPER, KD_GRIPPER
        elif any(x in joint_name.lower() for x in ["joint5", "joint6", "joint7", "wrist"]):
            return KP_WRIST, KD_WRIST
        else:
            # 根部关节 (Joint 1-4) 或其他
            return KP_BASE, KD_BASE

    def physics_loop(self):
        if glfw.window_should_close(self.window):
            rclpy.shutdown()
            return

        # PD 控制器
        for joint_name, act_id in self.actuator_map.items():
            if joint_name not in self.target_positions:
                continue
                
            target_q = self.target_positions[joint_name]
            
            # **修复**: 使用正确的索引
            qpos_addr = self.joint_qpos_map[joint_name]
            qvel_addr = self.joint_qvel_map[joint_name]
            
            current_q = self.data.qpos[qpos_addr]
            current_v = self.data.qvel[qvel_addr]  # 使用 qvel 地址
            
            # 获取 PD 参数
            kp, kd = self.get_pd_gains(joint_name)
            
            # 判断执行器类型
            is_position_actuator = "right_finger" in joint_name
            
            if is_position_actuator:
                self.data.ctrl[act_id] = target_q
            else:
                error = target_q - current_q
                torque = kp * error - kd * current_v
                self.data.ctrl[act_id] = torque
        
        # 物理步进
        for _ in range(2):
            mujoco.mj_step(self.model, self.data)

        # 渲染
        viewport = mujoco.MjrRect(0, 0, 1200, 900)
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, 
                              mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.mjr_context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

def main():
    rclpy.init()
    node = OpenArmDynamicsBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        glfw.terminate()
        node.destroy_node()

if __name__ == '__main__':
    main()