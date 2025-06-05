# 基于 Isaac Sim 和 ROS 2 Humble 的多机器人强化学习环境

本仓库提供了在 **Isaac Sim** 仿真环境和 **ROS 2 Humble** 导航栈基础上搭建的多机器人强化学习环境示例，主要用于研究和验证分布式 Proximal Policy Optimization（DPPO）在移动机器人导航任务中的应用。

## 功能特性

- **Isaac Sim 集成**：通过 `RL_scripts/env.py` 中的 `initialize_omniverse` 函数启动 Isaac Sim，并加载 `urdf/` 目录下的 USD/URDF 场景与机器人模型。
- **ROS 2 Humble**：环境依赖 ROS 2 Humble，并使用 nav2 堆栈进行定位与路径规划，相关参数在 `params/` 目录下提供。
- **多机器人支持**：`MultiRobotEnv` 类可同时管理多个机器人实例，负责状态采集、动作下发及奖励计算。
- **串口驱动示例**：`ros_serial.py` 提供了与实际机器人硬件通信的示例节点，可发布里程计、IMU 数据并接收速度控制指令。

## 依赖安装

1. 安装 [Isaac Sim](https://developer.nvidia.com/isaac-sim)（建议 2023 版本及以上），并确保其 ROS 2 Bridge 已启用。
2. 安装 ROS 2 Humble，配置好 `source /opt/ros/humble/setup.bash` 环境。
3. 克隆本仓库并安装 Python 依赖：

```bash
pip install -r requirements.txt
```

## 快速上手

以下代码片段展示了如何创建环境并与机器人交互：

```python
from RL_scripts.env import MultiRobotEnv

robot_names = ["saodi01", "saodi02", "saodi03"]
env = MultiRobotEnv(robot_names, "urdf/room.usd", history_len=5)

state, goal = env.reset("saodi01")
action = [0.1, 0.0]  # 前进指令
env.step(action, "saodi01")
```

仿真运行过程中，ROS 2 的导航组件将根据 nav2 参数文件控制机器人运动。更多细节可参考 `RL_scripts/env.py` 中的实现。

## 文件结构

- `RL_scripts/`：环境实现及数据集工具。
- `urdf/`：机器人和场景的 USD/URDF 文件。
- `params/`：nav2 相关参数配置。
- `maps/`、`rviz/`：示例地图和 RViz 配置文件。

## 许可

本项目仅供研究和学习使用，若在您的工作中使用本仓库，请注明来源。
