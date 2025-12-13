#!/usr/bin/env python3
import time
import os
import numpy as np
import mujoco

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker

class MujocoSimNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim_node')
        
        # --- 1. 加载 MuJoCo 模型 ---
        home_dir = os.path.expanduser("~")
        xml_path = os.path.join(home_dir, "arm_ws/src/arm/arm/scene/scene.xml")
        self.get_logger().info(f"正在加载模型: {xml_path}")
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # --- 2. 初始化 ROS 发布者 ---
        self.tf_broadcaster = TransformBroadcaster(self)
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)
        
        # --- 3. 设置仿真频率 (500Hz) ---
        self.timer = self.create_timer(0.002, self.timer_callback)
        self.get_logger().info("仿真节点已启动！")

    def timer_callback(self):
        # --- A. 物理步进 ---
        mujoco.mj_step(self.model, self.data)
        now = self.get_clock().now().to_msg()
        
        # ==========================================
        # 物品 1: 香蕉 (Banana)
        # ==========================================
        banana_pos = self.data.body('banana').xpos
        banana_quat = self.data.body('banana').xquat
        
        # 1. 发布 TF (World -> Banana)
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'world'
        t.child_frame_id = 'banana_link'
        t.transform.translation.x = banana_pos[0]
        t.transform.translation.y = banana_pos[1]
        t.transform.translation.z = banana_pos[2]
        t.transform.rotation.w = float(banana_quat[0])
        t.transform.rotation.x = float(banana_quat[1])
        t.transform.rotation.y = float(banana_quat[2])
        t.transform.rotation.z = float(banana_quat[3])
        self.tf_broadcaster.sendTransform(t)

        # 2. 发布 Marker (Banana)
        marker_b = Marker()
        marker_b.header.frame_id = "banana_link" # 跟随 TF
        marker_b.header.stamp = now
        marker_b.ns = "objects"
        marker_b.id = 0
        marker_b.type = Marker.MESH_RESOURCE
        marker_b.action = Marker.ADD
        marker_b.mesh_resource = "http://localhost:8000/textured.obj"
        marker_b.scale.x = 1.0; marker_b.scale.y = 1.0; marker_b.scale.z = 1.0
        
        # [修改点]：如果贴图加载失败，强制使用颜色覆盖，避免变黑
        marker_b.mesh_use_embedded_materials = False 
        marker_b.color.r = 1.0
        marker_b.color.g = 1.0
        marker_b.color.b = 0.0 # 黄色
        marker_b.color.a = 1.0
        
        self.marker_pub.publish(marker_b)

        # ==========================================
        # 物品 2: 桌子 (Table) - [新增部分]
        # ==========================================
        # 获取 MuJoCo 中 table body 的位置
        table_pos = self.data.body('table').xpos
        table_quat = self.data.body('table').xquat

        marker_t = Marker()
        # 注意：这里我们直接挂在 world 下，因为桌子是静态的
        marker_t.header.frame_id = "world" 
        marker_t.header.stamp = now
        marker_t.ns = "environment"
        marker_t.id = 1
        marker_t.type = Marker.CUBE # 使用基本几何体
        marker_t.action = Marker.ADD
        
        marker_t.pose.position.x = table_pos[0]
        marker_t.pose.position.y = table_pos[1]
        marker_t.pose.position.z = table_pos[2]
        marker_t.pose.orientation.w = float(table_quat[0])
        marker_t.pose.orientation.x = float(table_quat[1])
        marker_t.pose.orientation.y = float(table_quat[2])
        marker_t.pose.orientation.z = float(table_quat[3])

        marker_t.scale.x = 0.3 * 2
        marker_t.scale.y = 0.4 * 2
        
        # [修改点]：把这里从 0.4 改为 0.3，与 XML 保持一致
        marker_t.scale.z = 0.3 * 2  # 也就是 0.6 米高

        # 颜色：木头色
        marker_t.color.r = 0.6
        marker_t.color.g = 0.4
        marker_t.color.b = 0.2
        marker_t.color.a = 1.0

        self.marker_pub.publish(marker_t)

def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()