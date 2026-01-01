#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
import sys
import tf_transformations 
import numpy as np
import time

# ================= 配置区域 =================
# 感知模块输出的模拟坐标 (MuJoCo中的香蕉位置)
# ✅ 修改：X坐标从 0.5 改为 0.3，离底座更近
TARGET_BANANA_POS = [0.3, 0.0, 0.425] 

# 抓取时的姿态 (四元数) - 假设手掌垂直向下
GRASP_ORIENTATION = tf_transformations.quaternion_from_euler(0, 1.57, 0) 
# ===========================================

def main():
    rclpy.init()
    node = Node('auto_grasp_node')
    collision_object_publisher = node.create_publisher(CollisionObject, '/collision_object', 10)
    
    print("=== 🤖 系统初始化完成 ===")
    time.sleep(1)
    
    # --- 1. 添加桌子 ---
    table_co = CollisionObject()
    table_co.header.frame_id = "world" 
    table_co.id = "table"
    
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    # ✅ 修改：尺寸缩小。MuJoCo size="0.2 0.4 0.2" (半长) -> 这里对应全长 [0.4, 0.8, 0.4]
    box.dimensions = [0.4, 0.8, 0.4] 
    
    table_co.primitives = [box]
    table_co.primitive_poses = [PoseStamped().pose]
    # ✅ 修改：位置拉近到 0.3
    table_co.primitive_poses[0].position.x = 0.3
    table_co.primitive_poses[0].position.y = 0.0
    table_co.primitive_poses[0].position.z = 0.2
    table_co.operation = CollisionObject.ADD
    
    collision_object_publisher.publish(table_co)
    print("Published table collision object")
    
    # --- 2. 添加香蕉 ---
    banana_co = CollisionObject()
    banana_co.header.frame_id = "world"
    banana_co.id = "banana_collision"
    
    banana_box = SolidPrimitive()
    banana_box.type = SolidPrimitive.BOX
    banana_box.dimensions = [0.05, 0.15, 0.05]
    
    banana_co.primitives = [banana_box]
    banana_co.primitive_poses = [PoseStamped().pose]
    # ✅ 修改：位置拉近到 0.3 (与 TARGET_BANANA_POS 保持一致)
    banana_co.primitive_poses[0].position.x = TARGET_BANANA_POS[0]
    banana_co.primitive_poses[0].position.y = TARGET_BANANA_POS[1]
    banana_co.primitive_poses[0].position.z = TARGET_BANANA_POS[2]
    banana_co.operation = CollisionObject.ADD
    
    collision_object_publisher.publish(banana_co)
    print("Published banana collision object")
    
    print("=== 📦 环境障碍物已更新===")
    print("✅ 任务完成！")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()