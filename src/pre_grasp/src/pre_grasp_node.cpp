#include <memory>
#include <thread>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2/LinearMath/Quaternion.h>

// --- 任务参数配置 ---
// 香蕉的目标坐标
const double TARGET_X = 0.3;
const double TARGET_Y = 0.0;
const double TARGET_Z = 0.5;

// 预抓取高度偏移量 (单位: 米)
// 建议：初次测试给 0.2m，熟练后改为 0.15m 或更低
const double PRE_GRASP_OFFSET = 0.15; 

int main(int argc, char* argv[])
{
  // 1. 初始化节点
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("openarm_pre_grasp_controller");

  // 2. 启动多线程执行器 (Spin in a separate thread)
  // MoveIt 的 Action Client 需要在后台处理反馈，必须异步运行
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread([&executor]() { executor.spin(); }).detach();

  // 3. 初始化 MoveIt 接口
  // 注意："arm" 是 OpenArm SRDF 文件中定义的规划组名称
  // 如果报错 "Group not found"，请检查 openarm_moveit_config
  static const std::string PLANNING_GROUP = "left_arm";
  auto move_group = moveit::planning_interface::MoveGroupInterface(node, PLANNING_GROUP);

  // 4. 安全设置 (调试期必做!)
  move_group.setMaxVelocityScalingFactor(0.3); // 30% 速度
  move_group.setMaxAccelerationScalingFactor(0.3);
  move_group.setPoseReferenceFrame("base_link"); // 统一参考系

  RCLCPP_INFO(node->get_logger(), "=== OpenArm 预抓取任务开始 ===");

  // 5. 构建目标位姿
  geometry_msgs::msg::Pose target_pose;
  
  // A. 位置: 目标点 + Z轴抬升
  target_pose.position.x = TARGET_X;
  target_pose.position.y = TARGET_Y;
  target_pose.position.z = TARGET_Z + PRE_GRASP_OFFSET;

  // B. 姿态: 抓手垂直向下
  // 使用 TF2 库从欧拉角生成四元数
  // 假设: 绕 Y 轴旋转 90 度是垂直向下 (视具体 URDF 而定)
  tf2::Quaternion q;
  q.setRPY(M_PI, 0, M_PI / 2.0); // Roll, Pitch, Yaw
  target_pose.orientation.x = q.x();
  target_pose.orientation.y = q.y();
  target_pose.orientation.z = q.z();
  target_pose.orientation.w = q.w();

  // 打印日志方便调试
  RCLCPP_INFO(node->get_logger(), "目标坐标: x=%.2f, y=%.2f, z=%.2f (Offset applied)", 
              target_pose.position.x, target_pose.position.y, target_pose.position.z);

  // 6. 规划路径 (Plan)
  move_group.setPoseTarget(target_pose);
  
  moveit::planning_interface::MoveGroupInterface::Plan my_plan;
  bool success = (move_group.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);

  // 7. 执行路径 (Execute)
  if (success)
  {
    RCLCPP_INFO(node->get_logger(), "规划成功！正在移动机械臂...");
    move_group.execute(my_plan);
    RCLCPP_INFO(node->get_logger(), "任务完成：已到达预抓取点。");
    // 1. 定义夹爪的规划组名称
    // 注意：请确保你的 SRDF 文件中有一个名为 "left_gripper" 的 group
    static const std::string GRIPPER_GROUP = "left_gripper";
    auto gripper_move_group = moveit::planning_interface::MoveGroupInterface(node, GRIPPER_GROUP);

    // 2. 设置速度 (夹爪通常可以快一点)
    gripper_move_group.setMaxVelocityScalingFactor(1.0);
    gripper_move_group.setMaxAccelerationScalingFactor(1.0);

    // 3. 设置目标状态为 "open"
    // 注意：MoveIt 配置助手生成的配置中，通常会预设 "open" 和 "close" 姿态
    // 如果报错 "Unknown named target"，请检查你的 .srdf 文件中 <group_state name="open"> 的定义
    gripper_move_group.setNamedTarget("open");

    // 4. 规划并执行 (对于夹爪，直接用 move() 比较方便)
    RCLCPP_INFO(node->get_logger(), "正在打开夹爪...");
    moveit::core::MoveItErrorCode gripper_success = gripper_move_group.move();

    if (gripper_success == moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_INFO(node->get_logger(), "夹爪已打开");
        // 建议：给一点物理时间让气动/电动夹爪完全张开
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    } else {
        RCLCPP_ERROR(node->get_logger(), "夹爪打开失败！请检查 SRDF 中的 group_state 定义。");
    }
    // ==========================================
  }  
  else
  {
    RCLCPP_ERROR(node->get_logger(), "规划失败！可能原因：");
    RCLCPP_ERROR(node->get_logger(), "1. 目标点超出工作空间");
    RCLCPP_ERROR(node->get_logger(), "2. 姿态不可达 (IK解算失败)");
    RCLCPP_ERROR(node->get_logger(), "3. 自碰撞检测触发");
  }

  // 优雅退出
  rclcpp::shutdown();
  return 0;
}