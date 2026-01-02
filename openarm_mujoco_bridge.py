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

# PD 参数调优
# 核心修改：KP 从 20 改为 500。
# 20 太软了，接近目标时力矩几乎为0，所以关不紧也关不快。
KP_GRIPPER = 1000.0  
KD_GRIPPER = 5.0    # 稍微增加阻尼，防止KP太大导致震荡

# 模拟电机的最大力矩限制 (Nm)，防止飞出
MAX_TORQUE = 5.0 
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
        self.joint_qpos_map = {} 
        self.joint_qvel_map = {} 
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
                self.get_logger().info(f"发现执行器: {joint_name} (ID: {i})")

        # 3. 初始化目标
        mujoco.mj_forward(self.model, self.data)
        self.target_positions = {}
        
        # 4. 分类关节
        self.arm_joints = []      
        self.gripper_joints = []  
        
        for joint_name in self.joint_qpos_map.keys():
            if "finger" in joint_name.lower() or "gripper" in joint_name.lower():
                self.gripper_joints.append(joint_name)
            else:
                self.arm_joints.append(joint_name)
        
        self.get_logger().info(f"手臂关节: {len(self.arm_joints)} | 夹爪关节: {len(self.gripper_joints)}")

        # 5. ROS 订阅
        self.create_subscription(JointState, 'joint_states', self.ros_cb, 10)

        # 6. 图形界面初始化
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

        self.create_timer(0.01, self.physics_loop)

    def ros_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name in self.joint_qpos_map:
                self.target_positions[name] = pos

    def physics_loop(self):
        if glfw.window_should_close(self.window):
            rclpy.shutdown()
            return

        # --- 手臂逻辑 (Kinematic 直接位置控制) ---
        for joint_name in self.arm_joints:
            if joint_name in self.target_positions:
                qpos_addr = self.joint_qpos_map[joint_name]
                qvel_addr = self.joint_qvel_map[joint_name]
                self.data.qpos[qpos_addr] = self.target_positions[joint_name]
                self.data.qvel[qvel_addr] = 0.0

        # --- 夹爪逻辑 (Dynamics 力矩/PD控制) ---
        for joint_name in self.gripper_joints:
            if joint_name not in self.target_positions:
                continue
            
            if joint_name not in self.actuator_map:
                continue
                
            act_id = self.actuator_map[joint_name]
            target_q = self.target_positions[joint_name]
            
            # 读取当前状态
            qpos_addr = self.joint_qpos_map[joint_name]
            qvel_addr = self.joint_qvel_map[joint_name]
            current_q = self.data.qpos[qpos_addr]
            current_v = self.data.qvel[qvel_addr]
            
            # --- PD 控制计算 ---
            error = target_q - current_q
            
            # 计算力矩: P项 + D项
            torque = KP_GRIPPER * error - KD_GRIPPER * current_v
            
            # 力矩限幅 (Clamp)，防止数值爆炸
            torque = np.clip(torque, -MAX_TORQUE, MAX_TORQUE)
            
            # 写入控制量
            # 注意：如果你的XML里是<position> actuator，这个torque会被当作目标位置偏移
            # 如果是<motor> actuator，这才是真正的力矩。
            # 鉴于你说"无力"，说明你在用 motor 模式，或者 position gain 很小。
            self.data.ctrl[act_id] = torque 

        # 物理步进
        for _ in range(5): 
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