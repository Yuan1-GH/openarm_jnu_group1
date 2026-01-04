#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import mujoco
import mujoco.viewer  # 引入 viewer 模块
import numpy as np
import sys
import time

# ---------------- 配置区域 ----------------
MODEL_XML_PATH = "src/openarm_mujoco/v1/openarm_bimanual.xml"
KP_GRIPPER = 1000.0  
KD_GRIPPER = 5.0    
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

        # 2. 构建映射表 (保持原样)
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

        # 3. 初始化目标
        mujoco.mj_forward(self.model, self.data)
        self.target_positions = {}
        
        # 4. 分类关节 (保持原样)
        self.arm_joints = []      
        self.gripper_joints = []  
        for joint_name in self.joint_qpos_map.keys():
            if "finger" in joint_name.lower() or "gripper" in joint_name.lower():
                self.gripper_joints.append(joint_name)
            else:
                self.arm_joints.append(joint_name)

        # 5. ROS 订阅
        self.create_subscription(JointState, 'joint_states', self.ros_cb, 10)

        # 6. 【关键修改】启动被动式 Viewer (带有原生调试界面)
        # launch_passive 不会阻塞主线程，非常适合配合 ROS 的 spin
        self.viewer = mujoco.viewer.launch_passive(
            self.model, 
            self.data, 
            show_left_ui=True, 
            show_right_ui=True
        )
        
        # 设置初始视角 (可选)
        self.viewer.cam.lookat = np.array([0, 0, 0.5])
        self.viewer.cam.distance = 2.0
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -30

        self.get_logger().info("MuJoCo Viewer 启动成功，调试界面已加载。")

        # 启动物理循环定时器
        self.create_timer(0.01, self.physics_loop)

    def ros_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name in self.joint_qpos_map:
                self.target_positions[name] = pos

    def physics_loop(self):
        # 检查 Viewer 是否被用户关闭
        if not self.viewer.is_running():
            self.get_logger().info("Viewer closed by user, shutting down...")
            rclpy.shutdown()
            return

        # --- 手臂逻辑 (Kinematic) ---
        for joint_name in self.arm_joints:
            if joint_name in self.target_positions:
                qpos_addr = self.joint_qpos_map[joint_name]
                # qvel_addr = self.joint_qvel_map[joint_name] # Kinematic 模式通常不需要强制设 vel 为 0，除非你想完全锁死
                self.data.qpos[qpos_addr] = self.target_positions[joint_name]

        # --- 夹爪逻辑 (Dynamics PD) ---
        for joint_name in self.gripper_joints:
            if joint_name not in self.target_positions or joint_name not in self.actuator_map:
                continue
            
            act_id = self.actuator_map[joint_name]
            target_q = self.target_positions[joint_name]
            
            qpos_addr = self.joint_qpos_map[joint_name]
            qvel_addr = self.joint_qvel_map[joint_name]
            current_q = self.data.qpos[qpos_addr]
            current_v = self.data.qvel[qvel_addr]
            
            error = target_q - current_q
            torque = KP_GRIPPER * error - KD_GRIPPER * current_v
            torque = np.clip(torque, -MAX_TORQUE, MAX_TORQUE)
            self.data.ctrl[act_id] = torque 

        # 物理步进
        mujoco.mj_step(self.model, self.data)

        # 【关键修改】同步 Viewer
        # 这会把当前的 physics state 发送到 GUI 线程进行渲染
        self.viewer.sync()

def main():
    rclpy.init()
    node = OpenArmDynamicsBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 这里的清理也更简单了，只需要关闭 viewer
        if hasattr(node, 'viewer'):
            node.viewer.close()
        node.destroy_node()

if __name__ == '__main__':
    main()