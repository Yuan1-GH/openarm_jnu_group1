#include <memory>
#include <thread>
#include <vector>
#include <chrono>
#include <algorithm>

// ROS 2 核心
#include <rclcpp/rclcpp.hpp>
// ROS 2 Action 支持
#include <rclcpp_action/rclcpp_action.hpp>
// 夹爪控制消息类型
#include <control_msgs/action/gripper_command.hpp>

// MoveIt 2 接口
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

// 消息类型
#include <geometry_msgs/msg/pose.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>

// --- 用户配置区 ---
const double TARGET_X = 0.3;
const double TARGET_Y = 0.0;
const double TARGET_Z = 0.59; // 抓取高度

// 规划组名称
static const std::string ARM_GROUP = "left_arm";
static const std::string GRIPPER_GROUP = "left_gripper"; 

// 关键名称
static const std::string EE_LINK_NAME = "openarm_left_hand"; 
static const std::string OBJECT_ID = "banana_collision";

// 辅助高度
const double PRE_GRASP_HEIGHT = 0.15; 
const double LIFT_HEIGHT = 0.20;      

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("moveit_grasp_node");

  // 1. 启动多线程执行器（MoveIt 运行必需）
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread([&executor]() { executor.spin(); }).detach();

  RCLCPP_INFO(node->get_logger(), "========== 纯 MoveIt 抓取任务开始 (Action版) ==========");

  // 2. 初始化 MoveIt 接口
  auto arm_group = moveit::planning_interface::MoveGroupInterface(node, ARM_GROUP);
  auto gripper_group = moveit::planning_interface::MoveGroupInterface(node, GRIPPER_GROUP);
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

  // 设置规划参数
  arm_group.setMaxVelocityScalingFactor(0.3); 
  arm_group.setMaxAccelerationScalingFactor(0.3);
  arm_group.setPlanningTime(5.0); 
  arm_group.setGoalPositionTolerance(0.01);

  // ==========================================================
  // 阶段 I: 移动到预抓取点 (Pre-Grasp)
  // ==========================================================
  RCLCPP_INFO(node->get_logger(), "[Step 1] 前往预备点...");

  tf2::Quaternion q;
  q.setRPY(M_PI, 0, M_PI / 2.0); // 手爪垂直向下

  geometry_msgs::msg::Pose pre_grasp_pose;
  pre_grasp_pose.orientation.x = q.x();
  pre_grasp_pose.orientation.y = q.y();
  pre_grasp_pose.orientation.z = q.z();
  pre_grasp_pose.orientation.w = q.w();
  pre_grasp_pose.position.x = TARGET_X;
  pre_grasp_pose.position.y = TARGET_Y;
  pre_grasp_pose.position.z = TARGET_Z + PRE_GRASP_HEIGHT;

  arm_group.setPoseTarget(pre_grasp_pose);
  
  if (arm_group.move() == moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_INFO(node->get_logger(), "-> 到达预备点");
  } else {
      RCLCPP_ERROR(node->get_logger(), "预备点移动失败");
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // ==========================================================
  // 阶段 II: 张开夹爪 (Open)
  // ==========================================================
  RCLCPP_INFO(node->get_logger(), "[Step 2] 张开夹爪...");
  
  bool gripper_opened = false;
  auto named_targets = gripper_group.getNamedTargets();
  std::vector<std::string> open_names = {"open", "opened", "Open"};
  
  for (const auto& name : open_names) {
      if (std::find(named_targets.begin(), named_targets.end(), name) != named_targets.end()) {
          gripper_group.setNamedTarget(name);
          gripper_group.move();
          gripper_opened = true;
          RCLCPP_INFO(node->get_logger(), "-> 执行状态: %s", name.c_str());
          break;
      }
  }
  
  if (!gripper_opened) {
      RCLCPP_WARN(node->get_logger(), "-> 未找到命名状态，尝试手动设置关节值...");
      auto joint_names = gripper_group.getRobotModel()->getJointModelGroup(GRIPPER_GROUP)->getActiveJointModelNames();
      std::vector<double> joint_values(joint_names.size(), 0.04); 
      
      gripper_group.setJointValueTarget(joint_values);
      gripper_group.move();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // ==========================================================
  // 阶段 III: 直线路径下探 (Approach)
  // ==========================================================
  RCLCPP_INFO(node->get_logger(), "[Step 3] 直线下降...");

  geometry_msgs::msg::Pose current_pose = arm_group.getCurrentPose().pose;
  std::vector<geometry_msgs::msg::Pose> waypoints_down;
  geometry_msgs::msg::Pose target_pose = current_pose;
  target_pose.position.z = TARGET_Z; 
  waypoints_down.push_back(target_pose);

  moveit_msgs::msg::RobotTrajectory trajectory_down;
  double fraction = arm_group.computeCartesianPath(waypoints_down, 0.01, 0.0, trajectory_down, false);

  if (fraction > 0.3) {
      arm_group.execute(trajectory_down);
  } else {
      RCLCPP_ERROR(node->get_logger(), "下探规划失败 (覆盖率仅 %.2f)", fraction);
      return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // ==========================================================
  // 阶段 IV: 闭合夹爪 (Grasp) - 使用 Action Client
  // ==========================================================
  RCLCPP_INFO(node->get_logger(), "[Step 4] 通过 Action 闭合夹爪...");

  // 定义 Action 类型别名 (只定义 Action 消息类型)
  using GripperCommand = control_msgs::action::GripperCommand;
  // 注意：这里不需要定义 ClientT，直接在 create_client 中使用 GripperCommand

  // 创建 Action Client
  // 【关键修改】这里模板参数必须是 GripperCommand，不能是 Client<GripperCommand>
  auto gripper_action_client = rclcpp_action::create_client<GripperCommand>(node, "/left_gripper_controller/gripper_cmd");

  // 等待 Action Server 上线
  RCLCPP_INFO(node->get_logger(), "-> 等待夹爪 Action Server...");
  if (!gripper_action_client->wait_for_action_server(std::chrono::seconds(5))) {
    RCLCPP_ERROR(node->get_logger(), "Action server 不可用！");
  } else {
    // 构建目标
    auto goal_msg = GripperCommand::Goal();
    goal_msg.command.position = 0.02;
    goal_msg.command.max_effort = 5.0;

    // 发送目标
    RCLCPP_INFO(node->get_logger(), "-> 发送抓取指令 (Pos: 0.01, Effort: 5.0)");
    auto goal_handle_future = gripper_action_client->async_send_goal(goal_msg);

    // 等待请求被接受
    if (goal_handle_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        RCLCPP_ERROR(node->get_logger(), "发送目标超时");
    } else {
        auto goal_handle = goal_handle_future.get();
        if (!goal_handle) {
            RCLCPP_ERROR(node->get_logger(), "目标被服务器拒绝");
        } else {
            RCLCPP_INFO(node->get_logger(), "-> 指令已发送，开始等待 30 秒以确保完全闭合...");
            // 强制等待 30 秒
            std::this_thread::sleep_for(std::chrono::seconds(30));
        }
    }
  }

  // 逻辑处理：Attach 物体到末端
  moveit_msgs::msg::AttachedCollisionObject attached_object;
  attached_object.link_name = EE_LINK_NAME;   
  attached_object.object.header.frame_id = "world"; 
  attached_object.object.id = OBJECT_ID;
  attached_object.object.operation = attached_object.object.ADD;
  
  planning_scene_interface.applyAttachedCollisionObject(attached_object);
  RCLCPP_INFO(node->get_logger(), "-> 物体已在逻辑上吸附至末端");
  
  // ==========================================================
  // 阶段 V: 直线提升 (Lift)
  // ==========================================================
  RCLCPP_INFO(node->get_logger(), "[Step 5] 提升物体...");

  current_pose = arm_group.getCurrentPose().pose;
  std::vector<geometry_msgs::msg::Pose> waypoints_up;
  geometry_msgs::msg::Pose lift_pose = current_pose;
  lift_pose.position.z += LIFT_HEIGHT; 
  waypoints_up.push_back(lift_pose);

  moveit_msgs::msg::RobotTrajectory trajectory_up;
  fraction = arm_group.computeCartesianPath(waypoints_up, 0.01, 0.0, trajectory_up, false);

  if (fraction > 0.5) {
      arm_group.execute(trajectory_up);
      RCLCPP_INFO(node->get_logger(), "========== 抓取任务圆满完成 ==========");
  } else {
      RCLCPP_ERROR(node->get_logger(), "提升规划失败");
  }

  rclcpp::shutdown();
  return 0;
}