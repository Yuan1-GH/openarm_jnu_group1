#include <memory>
#include <thread>
#include <vector>
#include <chrono>
#include <algorithm>
#include <atomic>

// ROS 2 核心
#include <rclcpp/rclcpp.hpp>
// ROS 2 Action 支持
#include <rclcpp_action/rclcpp_action.hpp>
// 服务类型
#include <std_srvs/srv/set_bool.hpp>
// 夹爪控制消息类型
#include <control_msgs/action/gripper_command.hpp>

// MoveIt 2 接口
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

// 消息类型
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp> // Added
#include <tf2/LinearMath/Quaternion.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>

// 规划组名称
static const std::string ARM_GROUP = "left_arm";
static const std::string GRIPPER_GROUP = "left_gripper"; 

// 关键名称
static const std::string EE_LINK_NAME = "openarm_left_hand"; 
static const std::string OBJECT_ID = "banana_collision"; // MoveIt 中的碰撞体ID

// 辅助高度
const double PRE_GRASP_HEIGHT = 0.15; 
const double LIFT_HEIGHT = 0.20;      

class GraspNode : public rclcpp::Node {
public:
    GraspNode() : Node("moveit_grasp_node") {
        // 订阅感知结果
        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/detected_object_pose", 10, 
            std::bind(&GraspNode::pose_cb, this, std::placeholders::_1));
    }

    void pose_cb(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        if (!target_received_) {
            target_pose_ = msg->pose;
            target_received_ = true;
            RCLCPP_INFO(this->get_logger(), "收到目标位置: [%.3f, %.3f, %.3f]", 
                target_pose_.position.x, target_pose_.position.y, target_pose_.position.z);
        }
    }

    bool has_target() const { return target_received_; }
    geometry_msgs::msg::Pose get_target() const { return target_pose_; }

private:
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    geometry_msgs::msg::Pose target_pose_;
    std::atomic<bool> target_received_{false};
};

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<GraspNode>();

  // 1. 启动多线程执行器（MoveIt 运行必需）
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread([&executor]() { executor.spin(); }).detach();

  RCLCPP_INFO(node->get_logger(), "========== MoveIt + MuJoCo 视觉抓取任务开始 ==========");
  RCLCPP_INFO(node->get_logger(), "等待视觉感知节点发布目标位置...");

  // 等待目标
  while (rclcpp::ok() && !node->has_target()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  
  if (!rclcpp::ok()) return 0;

  auto target_pose = node->get_target();
  // 修正目标 Z (通常感知到的是中心，但我们需要抓取点，可能需要微调)
  // 假设感知的就是抓取中心，稍微太高/太低可能需要手动偏移，这里直接使用感知值
  double target_x = target_pose.position.x;
  double target_y = target_pose.position.y;
  double target_z = target_pose.position.z + 0.15; // 稍微抬高一点抓取点? 原始代码 target_z=0.62 while table=0.45. diff=0.17.
  // 之前的逻辑: TABLE=0.2(base)+0.2(size/2)=0.4top? NO.
  // AddCollisionBox: Table Z=0.2, Box dim=[..., 0.4]. Top surface = 0.2 + 0.4/2 = 0.4.
  // Banana pos Z = 0.425 (center). Top surface approx 0.45.
  // Old code TARGET_Z = 0.62. This seems very high. Maybe grasp approach point?
  // Let's assume the detected pose is the object center. We need to grasp slightly above center?
  // Or maybe the old code 0.62 was the *wrist* position?
  // The EE frame is usually at the wrist. The fingers extend down.
  // Let's try to match the old behavior:
  // Old logic: Target Z = 0.62. 
  // If object is at 0.425. Offset is ~0.2m.
  // So let's add 0.2m to the detected Z.
  target_z = target_pose.position.z + 0.175; 

  RCLCPP_INFO(node->get_logger(), "最终抓取目标 (Wrist): [%.3f, %.3f, %.3f]", target_x, target_y, target_z);

  // 2. 初始化 MoveIt 接口
  auto arm_group = moveit::planning_interface::MoveGroupInterface(node, ARM_GROUP);
  auto gripper_group = moveit::planning_interface::MoveGroupInterface(node, GRIPPER_GROUP);
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

  // 设置规划参数
  arm_group.setMaxVelocityScalingFactor(0.3); 
  arm_group.setMaxAccelerationScalingFactor(0.3);
  arm_group.setPlanningTime(5.0); 
  arm_group.setGoalPositionTolerance(0.01);

  // 初始化 MuJoCo 物理吸附服务客户端
  auto mujoco_client = node->create_client<std_srvs::srv::SetBool>("/mujoco_attach_object");

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
  pre_grasp_pose.position.x = target_x;
  pre_grasp_pose.position.y = target_y;
  pre_grasp_pose.position.z = target_z + PRE_GRASP_HEIGHT;

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
  geometry_msgs::msg::Pose down_pose = current_pose;
  down_pose.position.z = target_z; 
  waypoints_down.push_back(down_pose);

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
  // 阶段 IV: 闭合夹爪 (Grasp) - Action Client
  // ==========================================================
  RCLCPP_INFO(node->get_logger(), "[Step 4] 闭合夹爪...");

  using GripperCommand = control_msgs::action::GripperCommand;
  auto gripper_action_client = rclcpp_action::create_client<GripperCommand>(node, "/left_gripper_controller/gripper_cmd");

  if (!gripper_action_client->wait_for_action_server(std::chrono::seconds(5))) {
    RCLCPP_ERROR(node->get_logger(), "Action server 不可用！");
  } else {
    auto goal_msg = GripperCommand::Goal();
    goal_msg.command.position = 0.015; // 稍微闭紧一点以确保接触
    goal_msg.command.max_effort = 500.0;

    auto goal_handle_future = gripper_action_client->async_send_goal(goal_msg);

    if (goal_handle_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready) {
        auto goal_handle = goal_handle_future.get();
        if (goal_handle) {
            RCLCPP_INFO(node->get_logger(), "-> 抓取指令已发送，等待动作完成...");
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }
  }

  // ==========================================================
  // 逻辑 + 物理 双重吸附
  // ==========================================================
  
  // 1. MoveIt 逻辑吸附 (用于防碰撞)
  moveit_msgs::msg::AttachedCollisionObject attached_object;
  attached_object.link_name = EE_LINK_NAME;   
  attached_object.object.header.frame_id = "world"; 
  attached_object.object.id = OBJECT_ID;
  attached_object.object.operation = attached_object.object.ADD;
  planning_scene_interface.applyAttachedCollisionObject(attached_object);
  RCLCPP_INFO(node->get_logger(), "-> [MoveIt] 逻辑吸附完成");

  // 2. MuJoCo 物理吸附 (用于视觉仿真)
  if (mujoco_client->wait_for_service(std::chrono::seconds(2))) {
      auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
      request->data = true; // True = Attach

      auto future = mujoco_client->async_send_request(request);
      // 等待服务返回结果
      if (future.wait_for(std::chrono::seconds(2)) == std::future_status::ready) {
           auto result = future.get();
           if(result->success) {
               RCLCPP_INFO(node->get_logger(), "-> [MuJoCo] 物理吸附成功: %s", result->message.c_str());
           } else {
               RCLCPP_ERROR(node->get_logger(), "-> [MuJoCo] 物理吸附失败: %s", result->message.c_str());
           }
      }
  } else {
      RCLCPP_WARN(node->get_logger(), "-> [MuJoCo] 服务未在线，无法执行物理吸附");
  }

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
      RCLCPP_INFO(node->get_logger(), "========== 任务完成 ==========");
  } else {
      RCLCPP_ERROR(node->get_logger(), "提升规划失败");
  }

  rclcpp::shutdown();
  return 0;
}