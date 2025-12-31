#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import mujoco
import glfw
import numpy as np
import sys

# ---------------------------------------------------------
MODEL_XML_PATH = "src/openarm_mujoco/v1/openarm_bimanual.xml"
# ---------------------------------------------------------

class OpenArmMujocoBridge(Node):
    def __init__(self):
        super().__init__('openarm_mujoco_bridge')

        # 1. 加载 MuJoCo 模型
        try:
            self.get_logger().info(f"Loading MuJoCo model from: {MODEL_XML_PATH}")
            self.model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            self.get_logger().error(f"Failed to load MuJoCo model: {e}")
            sys.exit(1)

        # 2. 建立 MuJoCo 关节名称索引表
        self.mujoco_joint_map = {} 
        self.get_logger().info("MuJoCo Model Joints Found:")
        
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                qpos_addr = self.model.jnt_qposadr[i]
                self.mujoco_joint_map[name] = qpos_addr
                # 日志会显示实际加载的关节名，用于调试
                self.get_logger().info(f"   - ID: {i} | Name: {name} | Qpos Addr: {qpos_addr}")
        
        # 3. 初始化 ROS 2 订阅
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )
        
        # 4. 初始化图形界面 (GLFW)
        if not glfw.init():
            self.get_logger().error("GLFW init failed")
            sys.exit(1)
        
        self.window = glfw.create_window(1200, 900, "OpenArm Bimanual Simulation", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # 相机与场景
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        
        # 【修复点】：改名为 mjr_context，避免与 Node.context 冲突
        self.mjr_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        # 调整双臂视角
        self.cam.azimuth = 90
        self.cam.elevation = -20
        self.cam.distance = 2.5 
        self.cam.lookat = np.array([0.0, 0.0, 0.8])

        # 渲染循环
        self.create_timer(0.033, self.render_loop)

    def joint_state_callback(self, msg):
        """
        双臂核心逻辑：通过名称精确匹配
        """
        for ros_name, ros_pos in zip(msg.name, msg.position):
            if ros_name in self.mujoco_joint_map:
                addr = self.mujoco_joint_map[ros_name]
                self.data.qpos[addr] = ros_pos
            # 可以在这里加 else: print(ros_name) 来查看是否有未匹配的关节

    def render_loop(self):
        if glfw.window_should_close(self.window):
            rclpy.shutdown()
            return

        mujoco.mj_step(self.model, self.data) 
        
        viewport = mujoco.MjrRect(0, 0, 1200, 900)
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, 
                               mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
        
        # 【修复点】：使用 mjr_context
        mujoco.mjr_render(viewport, self.scene, self.mjr_context)

        glfw.swap_buffers(self.window)
        glfw.poll_events()

def main(args=None):
    rclpy.init(args=args)
    node = OpenArmMujocoBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 【修复点】：释放 mjr_context
        if hasattr(node, 'mjr_context'):
            mujoco.mjr_freeContext(node.mjr_context)
        glfw.terminate()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()