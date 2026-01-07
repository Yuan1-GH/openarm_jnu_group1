#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import cv2
import numpy as np
import message_filters # Need to sync rgb and depth

class SimplePerceptionNode(Node):
    def __init__(self):
        super().__init__('simple_perception_node')
        
        # 订阅图像 (RGB & Depth) 使用 ApproximateTimeSynchronizer
        self.img_sub = message_filters.Subscriber(self, Image, '/camera/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth_image')
        
        # Increase slop to 1.0s to tolerate jitter in simulation time vs wall time
        self.ts = message_filters.ApproximateTimeSynchronizer([self.img_sub, self.depth_sub], 10, 1.0)
        self.ts.registerCallback(self.sync_cb)

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
        
        self.get_logger().info("Perception Node Started. Waiting for RGB-D images...")

    def info_cb(self, msg):
        if self.camera_K is None:
            self.camera_K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f"Camera Info Received: K=\n{self.camera_K}")

    def sync_cb(self, rgb_msg, depth_msg):
        # self.get_logger().info("Sync callback triggered") 
        if self.camera_K is None:
            self.get_logger().warn("Skipping sync_cb: Camera Info not yet received")
            return

        # 1. 转换 RGB 图像
        img = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(rgb_msg.height, rgb_msg.width, 3)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # For visualization/OpenCV

        # 2. 转换 Depth 图像 (32FC1)
        depth_img = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width)

        # --- Visualization Change: Show Depth instead of Raw RGB ---
        # Normalize depth for display (0.0m to 2.0m -> 0-255)
        depth_vis = np.clip(depth_img, 0.0, 3.0) # Clip at 3m
        depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        
        cv2.imshow("Depth View", depth_vis)
        # -----------------------------------------------------------

        # 3. 颜色识别 (黄色香蕉)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 4. 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detection_img = img_bgr.copy() # Canvas for detection visualization

        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # 画图调试 (Separate Window 2)
                cv2.drawContours(detection_img, [c], -1, (0, 255, 0), 2)
                cv2.circle(detection_img, (cx, cy), 5, (0, 0, 255), -1)
                
                # --- Depth-Based Position Estimation ---
                
                # 获取该像素点的深度值 (Meters)
                # 注意边界检查
                if 0 <= cy < depth_img.shape[0] and 0 <= cx < depth_img.shape[1]:
                    z_val = depth_img[cy, cx]
                    self.get_logger().info(f"Center ({cx},{cy}) Depth: {z_val:.4f} m")
                    
                    if z_val > 0.1 and z_val < 5.0: # Valid depth range check
                        # 反投影 (De-projection)
                        # P_cam = Z * K_inv * [u, v, 1]
                        
                        fx = self.camera_K[0, 0]
                        fy = self.camera_K[1, 1]
                        cx_opt = self.camera_K[0, 2]
                        cy_opt = self.camera_K[1, 2]
                        
                        # Z is depth (forward axis in OpenCV frame)
                        Z = z_val
                        X = (cx - cx_opt) * Z / fx
                        Y = (cy - cy_opt) * Z / fy
                        
                        P_cam = np.array([X, Y, Z, 1.0])
                        
                        # Transform to World Frame
                        # P_world = T_wc_cv * P_cam
                        P_world = self.T_wc_cv @ P_cam
                        
                        # 发布 Pose
                        pose_msg = PoseStamped()
                        pose_msg.header.stamp = self.get_clock().now().to_msg()
                        pose_msg.header.frame_id = "world"
                        pose_msg.pose.position.x = P_world[0]
                        pose_msg.pose.position.y = P_world[1]
                        pose_msg.pose.position.z = P_world[2]
                        pose_msg.pose.orientation.w = 1.0
                        
                        self.pose_pub.publish(pose_msg)
                        
                        # 在图像上显示深度信息
                        cv2.putText(detection_img, f"Z: {Z:.3f}m", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 显示识别结果 (Separate Window 2)
        cv2.imshow("Object Detection", detection_img)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = SimplePerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
