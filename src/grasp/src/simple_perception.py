#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import cv2
import numpy as np
import tf_transformations

class SimplePerceptionNode(Node):
    def __init__(self):
        super().__init__('simple_perception_node')
        
        self.declare_parameter('target_z_height', 0.45) # 香蕉的大致中心高度
        self.target_z = self.get_parameter('target_z_height').value

        # 订阅图像
        self.img_sub = self.create_subscription(Image, '/camera/image_raw', self.img_cb, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera_info', self.info_cb, 10)
        
        # 发布位姿
        self.pose_pub = self.create_publisher(PoseStamped, '/detected_object_pose', 10)
        
        self.camera_model = None
        self.camera_K = None
        
        # 硬编码相机外参 (从 XML 获取: chest_camera)
        # Pos: -0.138 0.007 1.040
        # XYAxes: 0.010 -1.000 0.000 0.806 0.008 0.591
        
        # 1. 构建旋转矩阵 (MuJoCo Camera Frame: X-Right, Y-Up, Z-Back)
        x_axis = np.array([0.010, -1.000, 0.000])
        y_axis = np.array([0.806, 0.008, 0.591])
        
        # 归一化 (XML中的向量通常是归一化的，但为了保险)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Z = X cross Y
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        R_wc = np.column_stack((x_axis, y_axis, z_axis))
        
        # 2. 构建平移
        t_wc = np.array([-0.138, 0.007, 1.040])
        
        self.T_wc = np.eye(4)
        self.T_wc[:3, :3] = R_wc
        self.T_wc[:3, 3] = t_wc
        
        # 修正 MuJoCo Camera 坐标系到 OpenCV 坐标系 (Z-forward) 的差异
        # MuJoCo 的相机通常: X-Right, Y-Up, Z-Back (OpenGL style)
        # OpenCV 需要: X-Right, Y-Down, Z-Forward
        # 需要绕 X 轴旋转 180度 (翻转 Y 和 Z)
        R_gl_cv = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        
        self.T_wc_cv = self.T_wc @ R_gl_cv
        
        self.get_logger().info("Perception Node Started. Waiting for images...")

    def info_cb(self, msg):
        if self.camera_K is None:
            self.camera_K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f"Camera Info Received: K=\n{self.camera_K}")

    def img_cb(self, msg):
        if self.camera_K is None:
            return

        # 1. 转换图像
        # 假设 encoding="rgb8"
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        # RGB -> BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 2. 颜色识别 (黄色香蕉)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        # 黄色范围 (OpenCV H: 0-180)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 3. 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # 画图调试
                cv2.drawContours(img_bgr, [c], -1, (0, 255, 0), 2)
                cv2.circle(img_bgr, (cx, cy), 5, (0, 0, 255), -1)
                
                # 4. 2D -> 3D 投影 (Ray Casting)
                # 像素坐标 (u, v) -> 归一化平面 (x, y, 1)
                # P_cam = K_inv * [u, v, 1] * depth
                
                uv_hom = np.array([cx, cy, 1.0])
                K_inv = np.linalg.inv(self.camera_K)
                ray_cam = K_inv @ uv_hom # 射线方向 (在相机坐标系下)
                
                # 转换到世界坐标系方向
                # ray_world = R_wc_cv * ray_cam
                ray_world = self.T_wc_cv[:3, :3] @ ray_cam
                
                # 相机中心 (世界坐标)
                cam_origin = self.T_wc_cv[:3, 3]
                
                # 射线方程: P = O + t * D
                # 我们知道目标 Z = target_z
                # P.z = O.z + t * D.z => target_z = O.z + t * D.z
                # t = (target_z - O.z) / D.z
                
                if abs(ray_world[2]) > 1e-6:
                    t = (self.target_z - cam_origin[2]) / ray_world[2]
                    
                    if t > 0: # 物体在相机前方
                        p_world = cam_origin + t * ray_world
                        
                        # 发布 Pose
                        pose_msg = PoseStamped()
                        pose_msg.header.stamp = self.get_clock().now().to_msg()
                        pose_msg.header.frame_id = "world"
                        pose_msg.pose.position.x = p_world[0]
                        pose_msg.pose.position.y = p_world[1]
                        pose_msg.pose.position.z = p_world[2]
                        # 保持默认朝向 (或者根据物体主轴计算，这里简化)
                        pose_msg.pose.orientation.w = 1.0
                        
                        self.pose_pub.publish(pose_msg)
                        # self.get_logger().info(f"Detected at: {p_world}")

def main():
    rclpy.init()
    node = SimplePerceptionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
