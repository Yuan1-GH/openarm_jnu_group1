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
import tf_transformations # 需要安装 sudo apt install ros-humble-tf-transformations
import numpy as np
import time

# ================= 配置区域 =================
# 感知模块输出的模拟坐标 (MuJoCo中的香蕉位置)
TARGET_BANANA_POS = [0.5, 0.0, 0.45] 
# 抓取时的姿态 (四元数) - 假设手掌垂直向下 (根据OpenArm末端坐标系调整)
# 这里假设末端Z轴朝前，需要绕Y轴转90度让Z轴朝下
# 你可能需要根据实际手眼标定结果调整这里
GRASP_ORIENTATION = tf_transformations.quaternion_from_euler(0, 1.57, 0) 
# ===========================================

def main():
    # 1. 初始化 ROS 节点
    rclpy.init()
    
    # 创建一个简单的节点用于发布碰撞对象
    node = Node('auto_grasp_node')
    
    # 发布碰撞对象到规划场景
    collision_object_publisher = node.create_publisher(CollisionObject, '/collision_object', 10)
    
    print("=== 🤖 系统初始化完成 ===")
    
    # 等待发布者连接
    time.sleep(1)
    
    # 2. 添加环境障碍物到规划场景
    # 添加桌子
    table_co = CollisionObject()
    table_co.header.frame_id = "world"  # 或 "base_link"
    table_co.id = "table"
    
    # 创建桌子的几何形状
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [0.6, 0.8, 0.4]  # 长宽高
    
    # 设置桌子的位置
    table_co.primitives = [box]
    table_co.primitive_poses = [PoseStamped().pose]
    table_co.primitive_poses[0].position.x = 0.5
    table_co.primitive_poses[0].position.y = 0.0
    table_co.primitive_poses[0].position.z = 0.2
    table_co.operation = CollisionObject.ADD
    
    collision_object_publisher.publish(table_co)
    print("Published table collision object")
    
    # 添加香蕉
    banana_co = CollisionObject()
    banana_co.header.frame_id = "world"
    banana_co.id = "banana_collision"
    
    banana_box = SolidPrimitive()
    banana_box.type = SolidPrimitive.BOX
    banana_box.dimensions = [0.05, 0.15, 0.05]
    
    banana_co.primitives = [banana_box]
    banana_co.primitive_poses = [PoseStamped().pose]
    banana_co.primitive_poses[0].position.x = 0.5
    banana_co.primitive_poses[0].position.y = 0.0
    banana_co.primitive_poses[0].position.z = 0.45
    banana_co.operation = CollisionObject.ADD
    
    collision_object_publisher.publish(banana_co)
    print("Published banana collision object")
    
    print("=== 📦 环境障碍物已添加 ===")
    
    # 在ROS 2中，需要使用不同的方式来控制机械臂
    # 通常通过action接口或关节位置话题来控制
    print(">>> 由于ROS 2中API差异，此脚本需要额外的MoveIt 2接口")
    print(">>> 请使用ROS 2 MoveIt 2的Python接口或安装pymoveit2")
    
    # 延时以便观察
    time.sleep(2)
    
    print("✅ 任务完成！")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()