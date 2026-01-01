from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # 1. 加载 OpenArm 的 MoveIt 配置
    # 这里假设 MoveIt 配置包的名字是 "openarm_moveit_config"
    # 如果你的包名不同，请修改 package_name 参数
    # "openarm" 是 SRDF 中定义的机器人名称，通常不需要改
    moveit_config = MoveItConfigsBuilder("openarm", package_name="openarm_bimanual_moveit_config").to_moveit_configs()

    # 2. 定义节点
    # 将加载好的 robot_description (URDF) 和 robot_description_semantic (SRDF) 
    # 作为 parameters 传给我们的 C++ 节点
    pre_grasp_node = Node(
        package="pre_grasp",
        executable="pre_grasp_node",
        name="openarm_pre_grasp_controller",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            # 如果需要仿真时间，取消下面这行的注释
            # {"use_sim_time": True}, 
        ],
    )

    return LaunchDescription([
        pre_grasp_node
    ])