from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # =======================================================================
    # 关键点：这里必须填对你的 MoveIt 配置包名
    # 通常是 "你的机器人名" + "_moveit_config"
    # 如果你的 SRDF 文件在 "openarm_moveit_config" 包里，这里就填 "openarm"
    # =======================================================================
    moveit_config = MoveItConfigsBuilder("openarm", package_name="openarm_bimanual_moveit_config").to_moveit_configs()

    # 定义我们要启动的节点
    grasp_node = Node(
        package="grasp",  # 你的新包名
        executable="simple_grasp_node", # 你的可执行文件名
        output="screen",
        parameters=[
            moveit_config.to_dict(), # 重点：把机器人模型参数传进去
        ],
    )

    return LaunchDescription([
        grasp_node
    ])