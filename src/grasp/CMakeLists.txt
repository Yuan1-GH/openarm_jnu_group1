cmake_minimum_required(VERSION 3.8)
project(grasp) 
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED) # 包含 MoveGroupInterface, PlanningSceneInterface
find_package(moveit_core REQUIRED)                   # 包含 TrajectoryProcessing
find_package(geometry_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)             # 用于四元数转换
find_package(rclcpp_action REQUIRED)   
find_package(control_msgs REQUIRED)   
# --- 2. 定义可执行文件 ---
add_executable(simple_grasp_node src/simple_grasp.cpp)

# --- 3. 链接依赖 ---
ament_target_dependencies(simple_grasp_node
  rclcpp
  moveit_ros_planning_interface
  moveit_core
  geometry_msgs
  tf2_geometry_msgs
  rclcpp_action
  control_msgs
)

# --- 4. 安装规则 ---

install(TARGETS simple_grasp_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()