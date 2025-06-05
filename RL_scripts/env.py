import time
import threading
import signal
import sys
import subprocess
import numpy as np
import gym
from gym import spaces
import torch
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist
from queue import Queue
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import MarkerArray, Marker
from rclpy.node import Node
from rclpy.publisher import Publisher
from geometry_msgs.msg import Point
import random
import math
from geometry_msgs.msg import Quaternion
from rosgraph_msgs.msg import Clock
from collections import deque
import queue
import wandb

# Isaac Sim 相关导入
from omni.isaac.kit import SimulationApp
import carb
from isaac_ros2_messages.srv import SetPrimAttribute
import omni


kit = None
is_processing = False



class GoalMarkerPublisher(Node):
    def __init__(self, goal_positions):
        super().__init__('goal_marker_publisher')
        self.publisher = self.create_publisher(MarkerArray, '/goal_marker_array', 10)
        self.timer = self.create_timer(1.0, self.publish_goals)  # 每秒钟发布一次目标位置
        self.goal_positions = goal_positions
    def publish_goals(self):
        marker_array = MarkerArray()  # 创建 MarkerArray
        # 发布多个目标位置
        for i, (ns, goal) in enumerate(self.goal_positions.items()):
            marker = Marker()
            marker.header.frame_id = 'map'  # 目标在 "map" 坐标系中
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = ns  # 使用命名空间
            marker.id = i  # 设置唯一的 id
            marker.type = Marker.SPHERE  # 使用球体标记目标点
            marker.action = Marker.ADD  # 添加标记
            marker.scale.x = 0.1  # 设置目标球的大小
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = 1.0  # 红色标记
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0  # 完全不透明

            # 设置目标位置
            point = Point()
            point.x = goal[0]
            point.y = goal[1]
            point.z = 0.0  # 可以设置 z 轴为 0

            marker.pose.position = point
            marker_array.markers.append(marker)  # 将 Marker 添加到 MarkerArray

        # 发布 MarkerArray
        self.publisher.publish(marker_array)


class RewardCalculator:
    def __init__(self, obstacle_positions, goal_positions, collision_threshold=30, goal_threshold=0.3, gamma=0.99):
        """
        :param obstacle_positions: list of (x, y) tuples
        :param goal_positions: dict {namespace: (x, y)}
        :param collision_threshold: 判定碰撞的距离阈值
        :param goal_threshold: 判定到达目标的距离阈值
        """
        self.obstacle_positions = obstacle_positions
        self.goal_positions = goal_positions
        self.collision_threshold = collision_threshold
        self.goal_threshold = goal_threshold
        self.previous_linear_velocity = None
        self.previous_angular_velocity = None
        self.gamma = gamma  # 折扣因子


    def check_collision(self, pos, global_costmap, local_costmap, init_pose):
        """
        判断当前位置是否发生碰撞。
        使用 costmap 判断，假设机器人当前位置为 (x, y)，在 costmap 中检查其是否存在障碍物。
        :param pos: 当前位置 [x, y]
        :return: 是否碰撞（True/False）
        """
        x, y = pos

        #print(x_idx, y_idx)
        global_costmap = global_costmap.squeeze(0)  # 去掉多余的维度
        local_costmap = local_costmap.squeeze(0)  # 去掉多余的维度
        x_idx = local_costmap.shape[0]//2
        y_idx = local_costmap.shape[1]//2
        #print(f"global_costmap shape: {global_costmap.shape}, local_costmap shape: {local_costmap.shape}")
        # 检查位置是否越界
        if x_idx < 0 or y_idx < 0 or x_idx >= local_costmap.shape[0] or y_idx >= local_costmap.shape[1]:
            return True  # 如果越界，视为发生碰撞

        # 获取全局和局部 costmap 的值，检查是否超过碰撞阈值
        #print(local_costmap[x_idx, y_idx])
        if (local_costmap[x_idx, y_idx] > self.collision_threshold):
            return True  # 如果代价值超过阈值，表示发生碰撞
        
        return False  # 没有碰撞

    def check_goal_reached(self, robot_pos, goal_pos):
        """判断是否到达目标"""
        dist = np.linalg.norm(np.array(robot_pos[:2]) - np.array(goal_pos[:2]))
        return dist < self.goal_threshold

    def compute_reward(self, namespace, state, init_pose, count, action):
        """
        计算时序奖励
        :param namespace: 机器人 namespace
        :param state: dict {'stamps': [...], 'odoms': [...], 'local_costmaps': [...], 'global_costmaps': [...]}
        :param init_pose: 起始位姿 (x, y, theta)
        :param count: 当前步数（可用于阶段奖励）
        :return: reward, done, info
        """
        reward = {
            "dist_reward": 0.0,
            "collision_reward": 0.0,
            "goal_reward": 0.0,
            "dir_reward": 0.0,
            "acc_reward": 0.0,
            "vel_reward": 0.0,
            "reward_sum": 0.0,
        }

        robot_pos = np.array([s[:2] for s in state['odoms']])  # (seq_len, 2)
        theta = np.array([s[2] for s in state['odoms']])       # 朝向角 (seq_len,)
        velocities = np.linalg.norm(np.array([s[3:5] for s in state['odoms']]), axis=1)  # 线速度
        angular_velocities = np.array([s[5] for s in state['odoms']])  # 角速度
        goal_pos = np.array(self.goal_positions[namespace])

        # 修正位置：机器人相对于 init_pose 偏移

        # --- 距离相关奖励 ---
        dists = np.linalg.norm(robot_pos - goal_pos, axis=1)  # 每个时刻与目标点的距离
        progress = dists[:-1] - dists[1:]  # 距离的减少量（越大越好）
        discounts = self.gamma ** np.arange(len(progress))
        reward["dist_reward"] += np.sum(discounts * progress) * 6.0  # 距离进展奖励

        # --- 朝向奖励（cos(theta - 目标方向）越接近 1 越好）---
        if np.sum(progress) > 0.01:
            goal_vec = goal_pos - robot_pos
            goal_dir = np.arctan2(goal_vec[:, 1], goal_vec[:, 0])
            dir_cos_sim = np.cos(theta - goal_dir)
            reward["dir_reward"] += np.mean(dir_cos_sim) * 1.2  # 提高权重

        # --- 加速度惩罚（鼓励平滑）---
        linear_acc = np.abs(np.diff(velocities))
        angular_acc = np.abs(np.diff(angular_velocities))
        #reward["acc_reward"] -= (np.sum(linear_acc) * 0.5 + np.sum(angular_acc) * 0.2)

        # --- 速度奖励（越接近目标，速度奖励越高）---
        final_v = velocities[-1]
        final_d = dists[-1]
        speed_gain = 1 / (1 + np.exp(-10 * (0.5 - final_d)))  # 离目标越近，奖励越大
        reward["vel_reward"] += final_v * speed_gain * 0.8

        # --- 超速惩罚（防止冲得太猛）---
        if velocities.mean() > 0.4:
            reward["vel_reward"] -= 0.2 * velocities.mean()
        if np.abs(angular_velocities).mean() > 0.6:
            reward["vel_reward"] -= 0.2 * np.abs(angular_velocities).mean()

        # --- 碰撞惩罚 ---
        if state['local_costmaps'][-1].max() > 50:
            reward["collision_reward"] -= velocities[-1] * state['local_costmaps'][-1].max() * 0.0003

        # --- 终止条件 ---
        if final_d < 0.2:
            reward["goal_reward"] += 5.0
            done = True
            info = {'event': 'goal_reached'}
        elif self.check_collision(robot_pos[-1], state['global_costmaps'][-1], state['local_costmaps'][-1], init_pose):
            reward["collision_reward"] -= 0.01
            done = True
            info = {'event': 'collision'}
        else:
            done = False
            info = {}

        reward["reward_sum"] = sum(reward.values()) - reward["reward_sum"]  # 汇总
        return reward, done, info
    
    def update(self, ns, pose):
        self.goal_positions[ns] = pose



def initialize_omniverse(usd_file_path):
    """
    初始化 Isaac Sim，并加载指定的 USD 环境文件
    """
    global kit
    launch_config = {"headless": True}
    kit = SimulationApp(launch_config)
    
    # 启用 ROS 2 Bridge
    from omni.isaac.core.utils.extensions import enable_extension
    import omni.graph.core as og
    from omni.isaac.core.utils.stage import is_stage_loading
    enable_extension("omni.isaac.ros2_bridge")
    kit.update()
    enable_extension("omni.graph.core")
    kit.update()
    enable_extension("omni.graph.action")


    # 加载 USD 场景
    kit.update()
    omni.usd.get_context().open_stage(usd_file_path)
    kit.update()
    kit.update()
    print("Loading stage...")
    while is_stage_loading():
        simulation_app.update()
    print("Loading Complete")
    return kit

class RobotDataCollector(Node):
    def __init__(self, namespace, data_queue, history_len=5):
        """
        负责从 ROS 2 订阅机器人状态，并维护一个最新的状态信息队列。
        """
        super().__init__(f'{namespace}_collector')
        self.namespace = namespace
        self.data_queue = data_queue
        self.expert_action = [0.0, 0.0]
        self.manual_override = False
        self.last_human_cmd_time = 0.0


        self.history_len = history_len  # 记录历史帧数
        self.state_history = deque(maxlen=history_len)  # 存储过去 N 帧数据

        # 订阅代价地图、里程计和 amcl 位置
        self.create_subscription(OccupancyGrid, f'{namespace}/global_costmap/costmap', self.global_costmap_callback, 10)
        self.create_subscription(OccupancyGrid, f'{namespace}/local_costmap/costmap', self.local_costmap_callback, 10)
        self.create_subscription(Odometry, f'{namespace}/odom', self.odom_callback, 10)
        self.create_subscription(Twist,f'{namespace}/cmd_vel_manual',self.cmd_vel_callback,10)
            
        self.subscription = self.create_subscription(Clock,'/clock',self.clock_callback,10)

        # 发布 cmd_vel 话题
        self.cmd_vel_publisher = self.create_publisher(Twist, f'{namespace}/cmd_vel', 10)
        self.initial_pose_publisher = self.create_publisher(PoseWithCovarianceStamped, f'{namespace}/initialpose', 10)
        self.client = self.create_client(SetPrimAttribute, '/set_prim_attribute')

        # 初始化数据0
        self.global_costmap = None
        self.local_costmap = None
        self.odom = None
        self.amcl_pose = None
        self.current_time = None

    def global_costmap_callback(self, msg):
        global_costmap = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.global_costmap = np.rot90(global_costmap, k=-1).copy()
        #print(f"Robot {self.namespace} global costmap updated: {self.global_costmap.shape}")
    def cmd_vel_callback(self, msg):
        x = msg.linear.x
        z = msg.angular.z
        self.expert_action = [x, z]
        self.manual_override = True
        self.last_human_cmd_time = time.time()

    def get_expert_action(self):
        # 如果最近3秒内有人工控制输入，就使用人工控制指令
        if time.time() - self.last_human_cmd_time < 1.0:
            return self.expert_action
        else:
            self.manual_override = False
            return None  # 否则返回None，表示可以使用模型的动作

    def clock_callback(self, msg):
        self.current_time = msg.clock.sec + msg.clock.nanosec * 1e-9
        #print(f"Robot {self.namespace} clock updated: {self.current_time}")

    def local_costmap_callback(self, msg):
        local_costmap = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.local_costmap = np.rot90(local_costmap, k=-1).copy()
        #print(f"Robot {self.namespace} local costmap updated: {self.local_costmap.shape}")


    def quaternion_to_yaw(self,q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = self.quaternion_to_yaw(q)
        vel = msg.twist.twist.linear
        angular_vel = msg.twist.twist.angular
        self.odom = np.array([pos.x, pos.y, yaw, vel.x, vel.y, angular_vel.z])
        #print(f"Robot {self.namespace} odom updated: {self.odom}")

    def publish_initial_pose(self, position, yaw=0.0):
        def yaw_to_quaternion(yaw):
            """ 将 yaw 角转换为四元数 """
            return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = position[0]
        msg.pose.pose.position.y = position[1]
        msg.pose.pose.orientation = yaw_to_quaternion(yaw)  # 设置合法方向
        for _ in range(3):
            self.initial_pose_publisher.publish(msg)
            time.sleep(0.1)

    def get_closest_obstacle_distance(self, costmap, resolution, max_range=3.0):
        """
        costmap: np.ndarray, shape (H, W)
        resolution: float, meters per cell
        max_range: float, 最大感知距离，用于归一化

        return: 归一化后的最近障碍物距离，float ∈ [0.0, 1.0]
        """
        H, W = costmap.shape
        robot_pos = np.array([H // 2, W // 2])  # 假设机器人在地图中心
        indices = np.argwhere(costmap >= 50)
        if indices.shape[0] == 0:
            return 1.0  # 没有障碍，返回归一化后的最大距离

        # 计算最小距离（单位是米）
        distances = np.linalg.norm(indices - robot_pos, axis=1)
        min_dist = distances.min() * resolution

        # 归一化并裁剪到 [0.0, 1.0]
        norm_dist = min(min_dist / max_range, 1.0)
        return norm_dist
        
    def update_state(self,goal, init_pose):
        """获取当前机器人状态，如果数据尚未准备好，则返回 None"""
        if self.global_costmap is not None and self.local_costmap is not None and self.odom is not None and self.current_time is not None:
            # 归一化地图数据
            global_costmap_tensor = torch.tensor(self.global_costmap, dtype=torch.int8)
            local_costmap_tensor = torch.tensor(self.local_costmap, dtype=torch.int8)
            odom_s = self.odom
            odom_s[0] += init_pose[0]
            odom_s[1] += init_pose[1]
            odom_tensor = torch.tensor(odom_s, dtype=torch.float16)
            robot_pos = self.odom[:2]  # 机器人当前位置
            # 计算轨迹中每个时间步与目标点的距离
            init_pose = init_pose[:2] # 扩展 init_pose 的维度
            robot_pos = robot_pos + init_pose
            dist = torch.tensor((robot_pos - goal), dtype=torch.float16)
            obstacle_distance = torch.tensor([self.get_closest_obstacle_distance(self.local_costmap, 0.05)], dtype=torch.float16)  # 假设分辨率为 0.05 米
            goal = torch.tensor(goal, dtype=torch.float16)
            # 将代价地图和里程计数据拼接
            # 假设 odom 是一个 1D 向量，直接扩展为与 costmap 相同的维度来拼接

            # 拼接所有数据（global_costmap, local_costmap, odom）
            state = {
                "namespace": self.namespace,
                "stamp": self.current_time,
                "global_costmap": global_costmap_tensor,
                "local_costmap": local_costmap_tensor,
                "odom": odom_tensor,
                "dist": dist,
                "obstacle_distance": obstacle_distance,
                "goal": goal,
            }
            self.state_history.append(state)
            return (global_costmap_tensor.shape[0]*global_costmap_tensor.shape[1])*2 + odom_tensor.shape[0]

    def get_seq_state(self):
        if len(self.state_history) < self.history_len:
            #self.get_logger().warn("Not enough data in state history")
            return None  # 数据不足
        stamps = [s["stamp"] for s in self.state_history]
        if not all(stamps[i] <= stamps[i + 1] for i in range(len(stamps) - 1)):
            self.get_logger().warn(f"Time stamps are not in order, Stamps: {stamps}")
            return None
        
        global_costmaps = torch.stack([s["global_costmap"] for s in self.state_history])
        local_costmaps = torch.stack([s["local_costmap"] for s in self.state_history])
        odom_states = torch.stack([s["odom"] for s in self.state_history])
        dists = torch.stack([s["dist"].unsqueeze(0) for s in self.state_history])
        obstacle_distances = torch.stack([s["obstacle_distance"].unsqueeze(0) for s in self.state_history])
        goals = torch.stack([s["goal"] for s in self.state_history])
        #self.get_logger().info("Getting seq state")
        return {
            "namespace": self.namespace,
            "stamps": stamps,
            "global_costmaps": global_costmaps, # 形状 (seq_len, 1, H, W)
            "local_costmaps": local_costmaps,   # 形状 (seq_len, 1, H, W)
            "odoms": odom_states,  # 形状 (seq_len, features)
            "dists": dists,
            "obstacle_distances": obstacle_distances,
            "goals": goals,
        }

    def send_action(self, action):
        """发送控制指令"""
        msg = Twist()
        msg.linear.x = float(action[0])
        msg.angular.z = float(action[1])
        self.cmd_vel_publisher.publish(msg)


    def set_initial_pose(self, position):
        namespace = self.namespace
        request = SetPrimAttribute.Request()
        request.path = f'/World/{namespace}'
        request.attribute = 'xformOp:translate'
        request.value = str(position)  # Ensure it's a string
        
        try:
            # 异步调用
            response =  self.client.call(request)
        except Exception as e:
            self.get_logger().error(f"Error setting initial pose for {namespace}: {e}")
        
        """通过 ROS 2 客户端设置机器人的 baselink 初始位置"""
        # 创建请求对象
        request = SetPrimAttribute.Request()
        request.path = f'/World/{namespace}/base_link'  # 设置 baselink 的路径
        request.attribute = 'xformOp:translate'  # 需要设置的属性
        request.value = f"[0, 0, 0]"  # 设置 baselink 位置为字符串格式

        # 调用服务
        future = self.client.call(request)
        # 等待结果

        request = SetPrimAttribute.Request()
        request.path = f'/World/{namespace}/base_link'  # 设置 baselink 的路径
        request.attribute = 'xformOp:orient'  # 需要设置的属性
        request.value = f"[1, 0, 0, 0]"  # 设置 baselink 位置为字符串格式
         # 调用服务
        future = self.client.call(request)
        # 等待结果

pause = False

class MultiRobotEnv(gym.Env):
    def __init__(self, robot_namespaces, urdf_file_path, history_len):
        """
        自定义 Gym 环境，管理多个机器人的数据采集和控制
        """
        global stop, pause
        super(MultiRobotEnv, self).__init__()
        self.kit = initialize_omniverse(urdf_file_path)
        import omni.graph.core as og
        omni.timeline.get_timeline_interface().play()
        carb.settings.get_settings().set_int("/rtx/debugMaterialType", 0)
        carb.settings.get_settings().set("/app/show_developer_preference_section", True)
        def scheduler(signum, frame):
            global kit
            global is_processing
            if not is_processing and not pause:
                is_processing = True
                kit.update()  # 执行更新操作
                is_processing = False
        signal.signal(signal.SIGALRM, scheduler)  # 使用 SIGALRM 作为信号触发
        signal.setitimer(signal.ITIMER_REAL, 0.1, 0.1)  # 每 0.1 秒触发一次
        self.robot_namespaces = robot_namespaces
        self.history_len = history_len
        self.count = {'saodi01': 0,
                        'saodi02': 0,
                        'saodi03': 0}
        self.data_queue = Queue()
        self.goal_positions = {'saodi01': (2.0, 2.5),
                                'saodi02': (3.0, 0.0),
                                'saodi03': (1.5, -2.0)}
        self.initial_positions = {
            'saodi01': [0.0, 1.7, -0.7],
            'saodi02': [1.7, 1.5, -0.7],
            'saodi03': [2.0, 0.0, -0.7],
        }
        self.goal_marker_publisher = GoalMarkerPublisher(self.goal_positions)
        self.reward_calculator = RewardCalculator(
            obstacle_positions=[(1.5, 2.0), (3.0, 1.0)],
            goal_positions=self.goal_positions
        )

        # 机器人数据收集器
        self.history_len = 5
        self.robots = {ns: RobotDataCollector(ns, self.data_queue,self.history_len) for ns in robot_namespaces}
        self.state_dim = None

        # 线程管理 ROS 2
        self.executor = rclpy.executors.MultiThreadedExecutor()
        for robot in self.robots.values():
            self.executor.add_node(robot)
        self.executor.add_node(self.goal_marker_publisher)
        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()
        while not all([robot.get_seq_state() is not None for robot in self.robots.values()]):
            time.sleep(0.3)
            for robot in self.robots.values():
                self.state_dim = robot.update_state(goal=self.goal_positions[robot.namespace], init_pose=self.initial_positions[robot.namespace])
                print(f"Robot {robot.namespace} is Waiting Initial State")
        print("All robots are ready.")        


    def reset(self, ns):
        """
        重置单个机器人的初始位置并返回其初始状态
        """
        # 随机目标位置
        # x = random.uniform(-2.0, 2.0)
        # y = random.uniform(-2.0, 2.0)
        # self.goal_positions[ns] = (x, y)

        # 发布所有目标位置
        # self.goal_marker_publisher.publish_goals(self.goal_positions)

        # 设置初始位置
        self.robots[ns].state_history.clear()
        self.robots[ns].send_action([0.0, 0.0])  # 停止机器人
        self.robots[ns].set_initial_pose(self.initial_positions[ns])
        self.robots[ns].publish_initial_pose(self.initial_positions[ns])

        time.sleep(0.5)  # 等待初始位置发布完成

        # 获取状态
        while self.robots[ns].get_seq_state() is None:
            self.robots[ns].update_state(goal=self.goal_positions[ns], init_pose=self.initial_positions[ns])
            time.sleep(0.5)

        state = self.robots[ns].get_seq_state()

        return state, self.goal_positions[ns]

    def step(self, action, ns):
        """
        单个机器人 step 接口
        """
        # 发送动作
        self.robots[ns].send_action(action)

        # 等待状态更新
        time.sleep(0.1)

        # 获取所有状态
        self.robots[ns].update_state(goal=self.goal_positions[ns], init_pose=self.initial_positions[ns])
        next_state = self.robots[ns].get_seq_state()
        #print(f"Robot {ns} next state: {next_state}")




        if not next_state:
            print("No_Data_ERROR")
            done = True
            return None, -1, done, {}

        # 计算奖励
        reward, done, info = self.reward_calculator.compute_reward(ns, next_state, self.initial_positions[ns], self.count[ns], action)
        if info != {} and info is not None:
            print(f"Robot:{ns}, {info}")

        return next_state, reward, done, info

    def close(self):
        """关闭 ROS 2 节点"""
        global stop
        rclpy.shutdown()
        signal.setitimer(signal.ITIMER_REAL, 0)
        self.kit.close()

    def Pause(self):
        global pause
        pause = True
    
    def Play(self):
        global pause
        pause = False