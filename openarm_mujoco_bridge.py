#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool # 引入服务类型
import mujoco
import mujoco.viewer
import numpy as np
import sys
from scipy.spatial.transform import Rotation as R # 必须安装 scipy

# ---------------- 配置区域 ----------------
MODEL_XML_PATH = "src/openarm_mujoco/v1/openarm_bimanual.xml"
KP_GRIPPER = 1000.0  
KD_GRIPPER = 5.0    
MAX_TORQUE = 5.0 

# !!! 必须与 XML 中的名称一致 !!!
# 物体 Body 名称 (该 body 下必须有一个 type="free" 的 joint)
TARGET_OBJECT_NAME = "banana" 
# 夹爪末端 Body 名称 (将以此为基准计算相对位姿)
GRIPPER_LINK_NAME = "openarm_left_hand" 
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

        # 2. 构建关节和执行器映射
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

        # 3. 初始化控制相关变量
        mujoco.mj_forward(self.model, self.data)
        self.target_positions = {}
        
        self.arm_joints = []      
        self.gripper_joints = []  
        for joint_name in self.joint_qpos_map.keys():
            if "finger" in joint_name.lower() or "gripper" in joint_name.lower():
                self.gripper_joints.append(joint_name)
            else:
                self.arm_joints.append(joint_name)

        # 4. 获取物体 ID 用于物理绑定
        self.obj_found = False
        try:
            # 找到物体的 Body ID
            self.obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, TARGET_OBJECT_NAME)
            if self.obj_body_id == -1:
                raise ValueError(f"Body {TARGET_OBJECT_NAME} not found")
                
            # 找到物体关联的 Free Joint (用于设置 qpos)
            # body_jntadr[body_id] 返回该 body 下第一个 joint 的索引
            self.obj_jnt_id = self.model.body_jntadr[self.obj_body_id]
            if self.obj_jnt_id == -1:
                raise ValueError(f"Body {TARGET_OBJECT_NAME} has no joint (must be a free joint)")

            self.obj_qpos_adr = self.model.jnt_qposadr[self.obj_jnt_id]
            self.obj_vel_adr = self.model.jnt_dofadr[self.obj_jnt_id]
            
            # 找到夹爪的 Body ID
            self.gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_LINK_NAME)
            if self.gripper_body_id == -1:
                 raise ValueError(f"Gripper Body {GRIPPER_LINK_NAME} not found")

            self.obj_found = True
            self.get_logger().info(f"Target object '{TARGET_OBJECT_NAME}' initialized for physics attachment.")
        except Exception as e:
            self.get_logger().warn(f"Attachment setup failed: {e}. Simulation will run without object handling.")

        # 5. ROS 接口
        self.create_subscription(JointState, 'joint_states', self.ros_cb, 10)
        
        # 定义物理吸附服务
        self.create_service(SetBool, '/mujoco_attach_object', self.attach_callback)
        self.is_attached = False
        self.rel_pos = None  # 相对位置
        self.rel_quat = None # 相对旋转 (scipy Rotation object)

        # 6. 启动 Viewer
        self.viewer = mujoco.viewer.launch_passive(
            self.model, 
            self.data, 
            show_left_ui=True, 
            show_right_ui=True
        )
        
        # 初始视角
        self.viewer.cam.lookat = np.array([0, 0, 0.5])
        self.viewer.cam.distance = 2.0
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -30

        # 启动物理循环
        self.create_timer(0.01, self.physics_loop)

    def attach_callback(self, request, response):
        """ 服务回调：计算并锁定/解锁相对位姿 """
        if not self.obj_found:
            response.success = False
            response.message = "Object setup failed on startup"
            return response

        if request.data: # 请求吸附
            # 获取夹爪位姿 (World Frame)
            g_pos = self.data.xpos[self.gripper_body_id]
            g_quat = self.data.xquat[self.gripper_body_id] # [w, x, y, z]
            
            # 获取物体位姿 (World Frame, via qpos)
            # qpos 的前7位: [x, y, z, w, x, y, z]
            o_qpos = self.data.qpos[self.obj_qpos_adr : self.obj_qpos_adr+7]
            o_pos = o_qpos[0:3]
            o_quat = o_qpos[3:7] # [w, x, y, z]

            # 转换为 Scipy Rotation 对象 (注意 scipy 使用 [x, y, z, w])
            R_g = R.from_quat([g_quat[1], g_quat[2], g_quat[3], g_quat[0]])
            R_o = R.from_quat([o_quat[1], o_quat[2], o_quat[3], o_quat[0]])

            # 计算相对旋转: R_rel = R_g_inv * R_o
            self.rel_quat = R_g.inv() * R_o

            # 计算相对位置: P_rel = R_g_inv * (P_o - P_g)
            # 相当于在夹爪坐标系看物体的位置
            self.rel_pos = R_g.inv().apply(o_pos - g_pos)

            self.is_attached = True
            response.message = "Attached: Relative pose calculated"
            self.get_logger().info("Object ATTACHED.")
        else: # 请求释放
            self.is_attached = False
            response.message = "Detached"
            self.get_logger().info("Object DETACHED.")
            
        response.success = True
        return response

    def ros_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name in self.joint_qpos_map:
                self.target_positions[name] = pos

    def physics_loop(self):
        if not self.viewer.is_running():
            rclpy.shutdown()
            return

        # 1. 运动学控制：手臂
        for joint_name in self.arm_joints:
            if joint_name in self.target_positions:
                qpos_addr = self.joint_qpos_map[joint_name]
                self.data.qpos[qpos_addr] = self.target_positions[joint_name]

        # 2. 动力学控制 (PD)：夹爪
        for joint_name in self.gripper_joints:
            if joint_name in self.target_positions and joint_name in self.actuator_map:
                act_id = self.actuator_map[joint_name]
                target_q = self.target_positions[joint_name]
                qpos_addr = self.joint_qpos_map[joint_name]
                qvel_addr = self.joint_qvel_map[joint_name]
                
                curr_q = self.data.qpos[qpos_addr]
                curr_v = self.data.qvel[qvel_addr]
                
                torque = KP_GRIPPER * (target_q - curr_q) - KD_GRIPPER * curr_v
                torque = np.clip(torque, -MAX_TORQUE, MAX_TORQUE)
                self.data.ctrl[act_id] = torque 

        # 3. 物理吸附逻辑
        if self.is_attached and self.obj_found:
            # 获取当前夹爪位姿
            g_pos = self.data.xpos[self.gripper_body_id]
            g_quat = self.data.xquat[self.gripper_body_id]
            R_g = R.from_quat([g_quat[1], g_quat[2], g_quat[3], g_quat[0]])

            # 计算新的物体位置: P_new = P_g + R_g * P_rel
            new_o_pos = g_pos + R_g.apply(self.rel_pos)

            # 计算新的物体旋转: R_new = R_g * R_rel
            new_o_R = R_g * self.rel_quat
            new_o_quat = new_o_R.as_quat() # 返回 [x, y, z, w]

            # 强制覆盖物体 qpos
            # 位置
            self.data.qpos[self.obj_qpos_adr : self.obj_qpos_adr+3] = new_o_pos
            # 四元数 (MuJoCo 需要 w, x, y, z)
            self.data.qpos[self.obj_qpos_adr+3 : self.obj_qpos_adr+7] = [new_o_quat[3], new_o_quat[0], new_o_quat[1], new_o_quat[2]]

            # 消除速度，防止物理引擎产生爆炸性反作用力
            self.data.qvel[self.obj_vel_adr : self.obj_vel_adr+6] = 0.0

        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

def main():
    rclpy.init()
    node = OpenArmDynamicsBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'viewer'):
            node.viewer.close()
        node.destroy_node()

if __name__ == '__main__':
    main()