
  若需手动运行仿真和抓取演示，请在不同的终端窗口中
  分别执行以下命令（切记每个终端都需要先执行
  source install/setup.bash）：

  第一步：启动运动规划 (MoveIt 2)
  启动 MoveIt 并使用虚拟硬件接口 (Fake Hardware)
  进行轨迹规划。

     ros2 launch openarm_bimanual_moveit_config
     demo.launch.py use_fake_hardware:=true

  第二步：启动物理仿真 (MuJoCo Bridge)
  连接 ROS 2 话题与 MuJoCo
  物理引擎，并渲染相机数据。

     python3 openarm_mujoco_bridge.py
  > 注意: 该节点会发布 RGB 图像到
  /camera/image_raw 以及线性深度图到
  /camera/depth_image。

  第三步：启动感知节点
  检测目标物体（例如：黄色的香蕉）并利用同步的
  RGB-D 数据流计算其 3D 位姿。
  ```
  python3 src/grasp/src/simple_perception.py
```
  第四步：初始化环境
  向 MoveIt
  规划场景中添加碰撞物体（如桌子），以确保与
  MuJoCo 环境一致。
   ```
   python3 addCollisionBox.py
```
  第五步：执行抓取
  触发自主抓取序列。
  ```
   ros2 launch grasp run_grasp.launch.py
```
