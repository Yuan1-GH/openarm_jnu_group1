#include <memory>
#include <thread>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2/LinearMath/Quaternion.h>

// 关键头文件：用于给笛卡尔路径添加时间戳
#include <moveit/trajectory_processing/iterative_time_parameterization.h>

// --- 任务配置区 ---
// 目标物体的实际坐标 (香蕉的位置)
const double TARGET_X = 0.3;
const double TARGET_Y = 0.0;
const double TARGET_Z = 0.52;  // 修改1: 提高抓取点，避免插入桌子 (原0.505)

// 预备点相对于目标的高度
const double PRE_GRASP_HEIGHT = 0.15; 
// 抓取后提升的高度
const double LIFT_HEIGHT = 0.20;

// 定义规划组名称
static const std::string ARM_GROUP = "left_arm";
static const std::string GRIPPER_GROUP = "left_gripper";

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("simple_grasp_node");

  // 1. 启动多线程执行器
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread([&executor]() { executor.spin(); }).detach();

  // 2. 初始化规划组
  auto arm_group = moveit::planning_interface::MoveGroupInterface(node, ARM_GROUP);
  auto gripper_group = moveit::planning_interface::MoveGroupInterface(node, GRIPPER_GROUP);

  // 3. 安全与容差设置
  arm_group.setMaxVelocityScalingFactor(0.3); 
  arm_group.setMaxAccelerationScalingFactor(0.3);
  arm_group.setPlanningTime(5.0); 

  // [关键] 放宽容忍度，允许 MuJoCo 的重力下垂误差
  arm_group.setGoalPositionTolerance(0.02);    // 2cm
  arm_group.setGoalOrientationTolerance(0.05); // ~3度

  RCLCPP_INFO(node->get_logger(), "========== 任务开始 ==========");

  // ==========================================================
  // 阶段 I: 移动到预抓取点 (Pre-Grasp)
  // ==========================================================
  
  // 构建姿态 (垂直向下)
  tf2::Quaternion q;
  q.setRPY(M_PI, 0, M_PI / 2.0);

  geometry_msgs::msg::Pose pre_grasp_pose;
  pre_grasp_pose.orientation.x = q.x();
  pre_grasp_pose.orientation.y = q.y();
  pre_grasp_pose.orientation.z = q.z();
  pre_grasp_pose.orientation.w = q.w();
  pre_grasp_pose.position.x = TARGET_X;
  pre_grasp_pose.position.y = TARGET_Y;
  pre_grasp_pose.position.z = TARGET_Z + PRE_GRASP_HEIGHT;

  arm_group.setPoseTarget(pre_grasp_pose);
  
  RCLCPP_INFO(node->get_logger(), "[Step 1] 前往预抓取点 (%.3f, %.3f, %.3f)...", 
      TARGET_X, TARGET_Y, TARGET_Z + PRE_GRASP_HEIGHT);
  
  auto move_result = arm_group.move();
  if (move_result == moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_INFO(node->get_logger(), "-> 到达预抓取点。");
  } else {
      RCLCPP_WARN(node->get_logger(), "-> MoveIt 报告未完美到达，尝试强制继续...");
  }

  // 修改2: 添加等待，让机械臂完全稳定
  std::this_thread::sleep_for(std::chrono::milliseconds(800));

  // ==========================================================
  // 阶段 II: 张开夹爪 (Open)
  // ==========================================================
  RCLCPP_INFO(node->get_logger(), "[Step 2] 张开夹爪...");
  
  // 修改3: 先检查可用的命名状态
  auto named_targets = gripper_group.getNamedTargets();
  RCLCPP_INFO(node->get_logger(), "可用的夹爪状态:");
  for (const auto& target : named_targets) {
      RCLCPP_INFO(node->get_logger(), "  - %s", target.c_str());
  }
  
  // 尝试常见的命名状态
  bool gripper_opened = false;
  std::vector<std::string> open_names = {"open", "opened", "Open"};
  for (const auto& name : open_names) {
      if (std::find(named_targets.begin(), named_targets.end(), name) != named_targets.end()) {
          gripper_group.setNamedTarget(name);
          gripper_group.move();
          gripper_opened = true;
          RCLCPP_INFO(node->get_logger(), "-> 使用状态: %s", name.c_str());
          break;
      }
  }
  
  if (!gripper_opened) {
      RCLCPP_WARN(node->get_logger(), "-> 未找到'open'状态，尝试设置关节值...");
      // 备用方案：直接设置关节值 (假设夹爪是 prismatic joint)
      std::vector<double> joint_values = {0.04, 0.04}; // 4cm 开口，根据你的URDF调整
      gripper_group.setJointValueTarget(joint_values);
      gripper_group.move();
  }
  
  std::this_thread::sleep_for(std::chrono::milliseconds(500)); 

  // ==========================================================
  // 阶段 III: 直线路径下探 (Approach)
  // ==========================================================
  RCLCPP_INFO(node->get_logger(), "[Step 3] 直线下降...");

  arm_group.setStartStateToCurrentState();
  geometry_msgs::msg::Pose current_pose = arm_group.getCurrentPose().pose;
  
  RCLCPP_INFO(node->get_logger(), "当前位置: x=%.3f, y=%.3f, z=%.3f", 
      current_pose.position.x, current_pose.position.y, current_pose.position.z);
  RCLCPP_INFO(node->get_logger(), "目标位置: x=%.3f, y=%.3f, z=%.3f", 
      current_pose.position.x, current_pose.position.y, TARGET_Z);
  
  // 修改4: 使用当前的 x,y 坐标，只修改 z
  std::vector<geometry_msgs::msg::Pose> waypoints_down;
  geometry_msgs::msg::Pose target_pose = current_pose;  // 保留当前姿态和x,y
  target_pose.position.z = TARGET_Z;  // 只改变高度
  waypoints_down.push_back(target_pose);

  moveit_msgs::msg::RobotTrajectory trajectory_down;
  double fraction = arm_group.computeCartesianPath(
      waypoints_down, 
      0.02,   // 2cm步长
      0.0,    // 禁用跳跃检查
      trajectory_down,
      false   // 禁用碰撞检查
  );

  RCLCPP_INFO(node->get_logger(), "路径规划完成度: %.2f%%", fraction * 100);

  if (fraction > 0.3) {  // 修改5: 降低阈值到30%，强制执行部分路径
      robot_trajectory::RobotTrajectory rt(arm_group.getRobotModel(), ARM_GROUP);
      rt.setRobotTrajectoryMsg(*arm_group.getCurrentState(), trajectory_down);
      
      trajectory_processing::IterativeParabolicTimeParameterization iptp;
      bool success = iptp.computeTimeStamps(rt, 0.1, 0.1);

      if (success) {
          rt.getRobotTrajectoryMsg(trajectory_down);
          arm_group.execute(trajectory_down);
          if (fraction < 0.9) {
              RCLCPP_WARN(node->get_logger(), "-> 下探部分完成 (%.2f%%)", fraction * 100);
          } else {
              RCLCPP_INFO(node->get_logger(), "-> 下探完成！");
          }
      } else {
          RCLCPP_ERROR(node->get_logger(), "-> 时间参数化失败！");
      }
  } else {
      RCLCPP_ERROR(node->get_logger(), 
          "路径规划严重失败! fraction=%.2f%% - 尝试分段下降", fraction * 100);
      
      // 备用方案：分两段下降
      RCLCPP_WARN(node->get_logger(), "尝试备用方案：分段下降...");
      
      // 第一段：下降到中点
      std::vector<geometry_msgs::msg::Pose> waypoints_mid;
      geometry_msgs::msg::Pose mid_pose = current_pose;
      mid_pose.position.z = (current_pose.position.z + TARGET_Z) / 2.0;
      waypoints_mid.push_back(mid_pose);
      
      moveit_msgs::msg::RobotTrajectory traj_mid;
      double frac_mid = arm_group.computeCartesianPath(waypoints_mid, 0.02, 0.0, traj_mid, false);
      
      if (frac_mid > 0.5) {
          robot_trajectory::RobotTrajectory rt_mid(arm_group.getRobotModel(), ARM_GROUP);
          rt_mid.setRobotTrajectoryMsg(*arm_group.getCurrentState(), traj_mid);
          trajectory_processing::IterativeParabolicTimeParameterization iptp_mid;
          
          if (iptp_mid.computeTimeStamps(rt_mid, 0.1, 0.1)) {
              rt_mid.getRobotTrajectoryMsg(traj_mid);
              arm_group.execute(traj_mid);
              RCLCPP_INFO(node->get_logger(), "-> 第一段完成，继续下降...");
              std::this_thread::sleep_for(std::chrono::milliseconds(300));
              
              // 第二段：继续下降到目标
              arm_group.setStartStateToCurrentState();
              current_pose = arm_group.getCurrentPose().pose;
              
              std::vector<geometry_msgs::msg::Pose> waypoints_final;
              geometry_msgs::msg::Pose final_pose = current_pose;
              final_pose.position.z = TARGET_Z;
              waypoints_final.push_back(final_pose);
              
              moveit_msgs::msg::RobotTrajectory traj_final;
              double frac_final = arm_group.computeCartesianPath(waypoints_final, 0.02, 0.0, traj_final, false);
              
              if (frac_final > 0.3) {
                  robot_trajectory::RobotTrajectory rt_final(arm_group.getRobotModel(), ARM_GROUP);
                  rt_final.setRobotTrajectoryMsg(*arm_group.getCurrentState(), traj_final);
                  trajectory_processing::IterativeParabolicTimeParameterization iptp_final;
                  
                  if (iptp_final.computeTimeStamps(rt_final, 0.1, 0.1)) {
                      rt_final.getRobotTrajectoryMsg(traj_final);
                      arm_group.execute(traj_final);
                      RCLCPP_INFO(node->get_logger(), "-> 分段下探完成！");
                  }
              }
          }
      } else {
          RCLCPP_ERROR(node->get_logger(), "分段方案也失败，可能是关节限制问题");
          return 1;
      }
  }

  // 修改6: 再次等待稳定
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // ==========================================================
  // 阶段 IV: 闭合夹爪 (Grasp)
  // ==========================================================
  RCLCPP_INFO(node->get_logger(), "[Step 4] 闭合夹爪...");
  
  bool gripper_closed = false;
  std::vector<std::string> close_names = {"close", "closed", "Close", "grasp"};
  for (const auto& name : close_names) {
      if (std::find(named_targets.begin(), named_targets.end(), name) != named_targets.end()) {
          gripper_group.setNamedTarget(name);
          gripper_group.move();
          gripper_closed = true;
          RCLCPP_INFO(node->get_logger(), "-> 使用状态: %s", name.c_str());
          break;
      }
  }
  
  if (!gripper_closed) {
      RCLCPP_WARN(node->get_logger(), "-> 未找到'close'状态，设置关节值...");
      std::vector<double> joint_values = {0.01, 0.01}; // 1cm 闭合，根据你的URDF调整
      gripper_group.setJointValueTarget(joint_values);
      gripper_group.move();
  }
  
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // ==========================================================
  // 阶段 V: 直线提升 (Retract)
  // ==========================================================
  RCLCPP_INFO(node->get_logger(), "[Step 5] 提起物体...");

  arm_group.setStartStateToCurrentState();
  current_pose = arm_group.getCurrentPose().pose;
  
  RCLCPP_INFO(node->get_logger(), "提升起点: x=%.3f, y=%.3f, z=%.3f", 
      current_pose.position.x, current_pose.position.y, current_pose.position.z);
  
  std::vector<geometry_msgs::msg::Pose> waypoints_up;
  geometry_msgs::msg::Pose lift_pose = current_pose;
  lift_pose.position.z += LIFT_HEIGHT; 
  waypoints_up.push_back(lift_pose);

  RCLCPP_INFO(node->get_logger(), "提升终点: x=%.3f, y=%.3f, z=%.3f", 
      lift_pose.position.x, lift_pose.position.y, lift_pose.position.z);

  moveit_msgs::msg::RobotTrajectory trajectory_up;
  fraction = arm_group.computeCartesianPath(
      waypoints_up, 
      0.02,
      0.0,
      trajectory_up,
      false
  );

  RCLCPP_INFO(node->get_logger(), "提升路径完成度: %.2f%%", fraction * 100);

  if (fraction > 0.5) {
      robot_trajectory::RobotTrajectory rt_up(arm_group.getRobotModel(), ARM_GROUP);
      rt_up.setRobotTrajectoryMsg(*arm_group.getCurrentState(), trajectory_up);
      
      trajectory_processing::IterativeParabolicTimeParameterization iptp;
      if (iptp.computeTimeStamps(rt_up, 0.2, 0.2)) {
         rt_up.getRobotTrajectoryMsg(trajectory_up);
         auto exec_result = arm_group.execute(trajectory_up);
         
         if (exec_result == moveit::core::MoveItErrorCode::SUCCESS) {
             RCLCPP_INFO(node->get_logger(), "========== 任务圆满完成 ==========");
         } else {
             RCLCPP_WARN(node->get_logger(), "========== 提升执行异常，但任务基本完成 ==========");
         }
      } else {
          RCLCPP_ERROR(node->get_logger(), "-> 提升时间参数化失败");
      }
  } else {
      RCLCPP_ERROR(node->get_logger(), "-> 提升路径规划失败(%.2f%%)", fraction * 100);
  }

  rclcpp::shutdown();
  return 0;
}