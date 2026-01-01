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

# 夹爪 PD 参数（真实动力学模拟）
KP_GRIPPER = 200
KD_GRIPPER = 3
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

        # 2. 构建映射表
        self.joint_qpos_map = {}  # 名字 -> qpos 地址
        self.joint_qvel_map = {}  # 名字 -> qvel 地址
        self.actuator_map = {}
        
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.joint_qpos_map[name] = self.model.jnt_qposadr[i]
                self.joint_qvel_map[name] = self.model.jnt_dofadr[i]

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
        
        # 4. 分类关节：手臂关节（位置模拟）vs 夹爪关节（真实模拟）
        self.arm_joints = []      # 手臂关节列表
        self.gripper_joints = []  # 夹爪关节列表
        
        for joint_name in self.joint_qpos_map.keys():
            if "finger" in joint_name.lower():
                self.gripper_joints.append(joint_name)
            else:
                self.arm_joints.append(joint_name)
        
        self.get_logger().info(f"手臂关节（位置模拟）: {len(self.arm_joints)} 个")
        self.get_logger().info(f"夹爪关节（真实模拟）: {len(self.gripper_joints)} 个")

        # 5. ROS 订阅
        self.create_subscription(JointState, 'joint_states', self.ros_cb, 10)

        # 6. 图形界面
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

        # 7. 物理循环
        self.create_timer(0.01, self.physics_loop)

    def ros_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name in self.joint_qpos_map:
                self.target_positions[name] = pos

    def physics_loop(self):
        if glfw.window_should_close(self.window):
            rclpy.shutdown()
            return

        # ========== 手臂部分：位置模拟（直接设置位置） ==========
        for joint_name in self.arm_joints:
            if joint_name not in self.target_positions:
                continue
            
            qpos_addr = self.joint_qpos_map[joint_name]
            qvel_addr = self.joint_qvel_map[joint_name]
            
            # 直接设置位置，速度清零（kinematic 模式）
            self.data.qpos[qpos_addr] = self.target_positions[joint_name]
            self.data.qvel[qvel_addr] = 0.0

        # ========== 夹爪部分：真实动力学模拟（PD 控制） ==========
        for joint_name in self.gripper_joints:
            if joint_name not in self.target_positions:
                continue
            
            if joint_name not in self.actuator_map:
                continue
                
            act_id = self.actuator_map[joint_name]
            target_q = self.target_positions[joint_name]
            
            qpos_addr = self.joint_qpos_map[joint_name]
            qvel_addr = self.joint_qvel_map[joint_name]
            
            current_q = self.data.qpos[qpos_addr]
            current_v = self.data.qvel[qvel_addr]
            
            # 判断是否为位置执行器
            is_position_actuator = "right_finger" in joint_name
            
            if is_position_actuator:
                # 位置执行器直接设置目标位置
                self.data.ctrl[act_id] = target_q
            else:
                # 力矩执行器使用 PD 控制
                error = target_q - current_q
                torque = KP_GRIPPER * error - KD_GRIPPER * current_v
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