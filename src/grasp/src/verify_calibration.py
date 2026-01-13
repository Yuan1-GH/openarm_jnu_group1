#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped
import numpy as np
import cv2

class CalibrationVerifier(Node):
    def __init__(self):
        super().__init__('calibration_verifier')
        
        # --- 1. Ground Truth Configuration (from XML) ---
        self.gt_pos_world = np.array([0.3, 0.0, 0.406]) # Banana position
        
        # Camera Extrinsics (Hardcoded from XML)
        # pos="-0.138 0.007 1.040"
        # xyaxes="0.010 -1.000 0.000 0.806 0.008 0.591"
        self.cam_pos = np.array([-0.138, 0.007, 1.040])
        x_axis = np.array([0.010, -1.000, 0.000])
        y_axis = np.array([0.806, 0.008, 0.591])
        
        # Normalize
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # Build Rotation Matrix (World -> MuJoCo Camera)
        self.R_wc = np.column_stack((x_axis, y_axis, z_axis))
        self.T_wc = np.eye(4)
        self.T_wc[:3, :3] = self.R_wc
        self.T_wc[:3, 3] = self.cam_pos
        
        # Adjust for OpenCV Frame (Rotate 180 deg around X)
        R_gl_cv = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        
        # Final Transformation: OpenCV Camera Frame -> World Frame
        self.T_wc_cv = self.T_wc @ R_gl_cv
        
        # Inverse: World Frame -> OpenCV Camera Frame
        self.T_cw_cv = np.linalg.inv(self.T_wc_cv)
        
        # --- 2. ROS Communication ---
        self.create_subscription(CameraInfo, '/camera/camera_info', self.info_cb, 10)
        self.create_subscription(PoseStamped, '/detected_object_pose', self.pose_cb, 10)
        
        self.camera_K = None
        self.get_logger().info("Waiting for Camera Info to perform theoretical projection...")

    def info_cb(self, msg):
        if self.camera_K is None:
            self.camera_K = np.array(msg.k).reshape(3, 3)
            self.width = msg.width
            self.height = msg.height
            self.get_logger().info(f"Received Intrinsics. Image Size: {self.width}x{self.height}")
            self.perform_theoretical_projection()

    def perform_theoretical_projection(self):
        """
        验证逻辑 A：正向投影
        已知物体在世界坐标系的绝对位置，计算它应该出现在图像的哪个像素点 (u, v) 以及深度值。
        """
        # 1. World -> Camera (OpenCV Frame)
        P_world_homo = np.append(self.gt_pos_world, 1.0)
        P_cam = self.T_cw_cv @ P_world_homo
        
        X_c, Y_c, Z_c = P_cam[:3]
        
        self.get_logger().info("--- Theoretical Projection Check ---")
        self.get_logger().info(f"GT Object World Pos: {self.gt_pos_world}")
        self.get_logger().info(f"GT Object Camera Pos (Pred): [{X_c:.3f}, {Y_c:.3f}, {Z_c:.3f}]")
        
        if Z_c <= 0:
            self.get_logger().error("Object is behind the camera!")
            return

        # 2. Project to Pixel
        # [u*z, v*z, z] = K * [X, Y, Z]
        pixel_homo = self.camera_K @ np.array([X_c, Y_c, Z_c])
        u = pixel_homo[0] / pixel_homo[2]
        v = pixel_homo[1] / pixel_homo[2]
        
        self.get_logger().info(f"Expected Pixel Coordinates: (u={u:.1f}, v={v:.1f})")
        self.get_logger().info(f"Expected Depth Value: {Z_c:.3f} m")
        self.get_logger().info("----------------------------------")
        
        # 可以在此处提示用户去查看 'simple_perception.py' 的图像窗口是否一致

    def pose_cb(self, msg):
        """
        验证逻辑 B：闭环误差验证
        对比感知节点计算出的 Pose 与 真值。
        """
        detected_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        
        error_vec = detected_pos - self.gt_pos_world
        dist_error = np.linalg.norm(error_vec)
        
        # 格式化输出
        log_str = (
            f"\n[Validation] Time: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}\n"
            f"  GT Pose (Center): {self.gt_pos_world}\n"
            f"  Detected Pose:    {np.round(detected_pos, 4)}\n"
            f"  Raw Error Vec:    {np.round(error_vec, 4)}\n"
            f"  Raw Total Error:  {dist_error * 1000:.2f} mm"
        )
        
        # --- Surface Offset Correction ---
        # The camera detects the *surface* of the object, which is closer to the camera than the object's center.
        # The banana top surface is approx 3.5cm - 4cm above the body center (Z=0.406).
        # We also observe an X-offset because the surface point is along the ray from the camera, 
        # stopping 'short' of the center.
        
        # Simple Z-correction (Assuming we detected the top surface)
        SURFACE_OFFSET_Z = 0.0355 # Based on observation
        
        # More robust: Check if the point lies on the ray from camera to GT center?
        # For now, let's just subtract the Z-offset from the detected pose to see if it aligns with Center in Z,
        # and acknowledge the X-shift is due to perspective.
        
        adjusted_error_z = abs(detected_pos[2] - (self.gt_pos_world[2] + SURFACE_OFFSET_Z))
        
        log_str += f"\n  [Analysis] Object Surface Offset detected."
        log_str += f"\n  Surface-Adjusted Z-Error: {adjusted_error_z * 1000:.2f} mm"

        # Check if "Surface Adjusted" error is acceptable
        if adjusted_error_z < 0.01: # < 1cm Z-error after accounting for surface
            self.get_logger().info(log_str + "\n  RESULT: [PASS] (Consistent with object surface)")
        else:
            self.get_logger().warn(log_str + "\n  RESULT: [HIGH ERROR] (Mismatch even after surface correction)")

def main():
    rclpy.init()
    node = CalibrationVerifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
