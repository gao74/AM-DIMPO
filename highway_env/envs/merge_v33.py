import time

import numpy as np
from gym.envs.registration import register
from typing import Tuple
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.road.objects import Obstacle
from highway_env.vehicle.kinematics import Vehicle
from scipy.spatial import KDTree

class MergeEnv(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 10
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True
                #"steering_range": [-1, 1]
            },
            "controlled_vehicles": 1,
            "screen_width": 800,
            "screen_height": 120,
            "centering_position": [0.3, 0.5],
            "scaling": 3,
            "simulation_frequency": 15,
            "duration": 50,
            "policy_frequency": 5,
            "reward_speed_range": [5, 20],
            "COLLISION_REWARD": 15,
            "HIGH_SPEED_REWARD": 4,
            "HEADWAY_COST": 4,
            "HEADWAY_TIME": 1.2,
            "MERGING_LANE_COST": 5,
            "traffic_density": 4,
            "show_trajectories": True,
            "render_agent_indicators": True
        })
        return config

    def _reward(self, action: int) -> float:
        # 直接返回单个受控车辆的奖励
        return self._agent_reward(action, self.controlled_vehicles[0])

    def _compute_headway_distance(self, vehicle):
        positions = np.array([v.position for v in self.road.vehicles])
        kdtree = KDTree(positions)
        neighbors = kdtree.query_ball_point(vehicle.position, 80)
        min_distance = 150
        # 定义目标车道
        target_lanes = []
        if vehicle.lane_index == ("b", "c", 1):  # 假设 "bc1" 是 ("b", "c", 1)
            target_lanes = [("b", "c", 0), ("c", "d", 0)]  # "bc0" 和 "cd0"

        for idx in neighbors:
            v = self.road.vehicles[idx]
            if v is not vehicle:
                # 如果车辆在 "bc1" 上，考虑目标车道；否则只考虑同一车道
                if (target_lanes and v.lane_index in target_lanes) or (
                        not target_lanes and v.lane_index == vehicle.lane_index):
                    if v.position[0] > vehicle.position[0]:
                        distance = v.position[0] - vehicle.position[0]
                        min_distance = min(min_distance, distance)
        return min_distance

    def _check_lane_and_angle(self, vehicle: Vehicle) -> bool:
        is_on_target_lane = vehicle.lane_index in [("b", "c", 0), ("c", "d", 0)]
        angle_within_range = -0.15 <= vehicle.heading <= 0.15
        return is_on_target_lane and angle_within_range

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:

        cd0_end = sum(self.ends)  # cd0 车道末端位置
        scaled_speed = utils.lmap(vehicle.speed, [5, 20], [0, 1])
        print(f"Current lane: {vehicle.lane_index}")

        # 计算后车距离的辅助函数
        def calculate_backward_distance(vehicle, target_lanes=None):
            backward_distance = 300  # 初始大值
            for v in self.road.vehicles:  # 确保只遍历车辆对象（非障碍物）
                if not hasattr(v, 'lane_index'):  # 跳过没有车道索引的对象（如障碍物）
                    continue
                if v is not vehicle:
                    # 检查车道是否匹配（目标车道或同车道）
                    in_target_lanes = target_lanes and v.lane_index in target_lanes
                    same_lane = not target_lanes and v.lane_index == vehicle.lane_index
                    if in_target_lanes or same_lane:
                        if v.position[0] < vehicle.position[0]:  # 后车
                            distance = vehicle.position[0] - v.position[0]
                            if distance < backward_distance:
                                backward_distance = distance
            return backward_distance if backward_distance < 300 else None  # 返回有效距离或None

        # 车头时距奖励（已有逻辑）
        headway_distance = self._compute_headway_distance(vehicle)
        if vehicle.lane_index in [("b", "c", 0), ("c", "d", 0)] and vehicle.speed > 0:
            print("headway_distance：", headway_distance)
            print("vehicle.speed：", vehicle.speed)
            Headway_cost = np.log(headway_distance / (1 * vehicle.speed))
            print("np.log(headway_distance / (1 * vehicle.speed))：", np.log(headway_distance / (1 * vehicle.speed)))
        else:
            Headway_cost = 0
        safety_bonus = 1 if Headway_cost > 0 else 0
        headway_cost_reward = 2 + 2 * (Headway_cost if Headway_cost < 0 else 0)

        reward_backward = 0
        has_not_forward_car = False
        if vehicle.lane_index in [("b", "c", 0), ("c", "d", 0)] and vehicle.speed > 0:
            # 检查是否有前车
            if headway_distance > 100:  # 假设无前车
                print("无前车，有后车")
                has_not_forward_car = True
                Headway_cost = 0
                backward_distance = calculate_backward_distance(vehicle, [("b", "c", 0), ("c", "d", 0)])
                if backward_distance is not None:
                    reward_backward = backward_distance / 20 - 1 # 后车距离越远，奖励越高（示例缩放）
                    print("后车距离奖励：",reward_backward)
                else:
                    reward_backward = 0  # 无后车时的固定奖励
            Headway_cost = np.log(headway_distance / (1 * vehicle.speed))
            print("np.log(headway_distance / (1 * vehicle.speed))：", np.log(headway_distance / (1 * vehicle.speed)))

        alignment_bonus = 0

        if (vehicle.lane_index == ("b", "c", 0) or vehicle.lane_index == ("c", "d", 0)):
            reward = 0
            #1.位置对齐奖励
            alignment_bonus += 0.8

            #2.角度惩罚
            steering_penalty = 0
            steering_penalty = -4 * max(0, abs(action[1]) - 0.25)
            if vehicle.heading == 0:
                steering_penalty += 0.8
            if vehicle.heading <= 0.2 and vehicle.heading>=-0.2:
                steering_penalty += 0.5

            # 3.车头时距惩罚
            headway_penalty = 0
            if vehicle.lane_index in [("b", "c", 0), ("c", "d", 0)] and vehicle.speed > 0 and has_not_forward_car==False:
                print("vehicle.speed:",vehicle.speed)
                min_safe_distance = vehicle.speed * 1  # 1 秒车头时距
                max_safe_distance = vehicle.speed * 1.75  # 3 秒车头时距
                if headway_distance < min_safe_distance:
                    headway_penalty = (headway_distance - min_safe_distance) * 0.2
                if headway_distance > max_safe_distance:
                    headway_penalty = (max_safe_distance - headway_distance) * 0.2
                if headway_distance >= min_safe_distance and headway_distance <= max_safe_distance:
                    headway_penalty = (max_safe_distance - min_safe_distance) * 0.2
                print(
                    f"headway_penalty: {headway_penalty}, min_safe: {min_safe_distance}, max_safe: {max_safe_distance}")
            elif has_not_forward_car==True:
                headway_penalty = 0

            # 4.倒退惩罚
            backward_penalty = 0
            if not hasattr(vehicle, 'last_x_position'):
                vehicle.last_x_position = vehicle.position[0]  # 初次设置上一步位置
            else:
                delta_x = vehicle.position[0] - vehicle.last_x_position  # 计算 x 坐标变化
                backward_penalty = 5 * delta_x - 1
                # if delta_x < 0:
                #     backward_penalty = -20 * abs(delta_x)  # 倒退惩罚：倒退距离 * 5
                # if delta_x ==0:
                #     backward_penalty = -1 # 倒退惩罚：倒退距离 * 5
                # if delta_x >=0 and delta_x<=0.5:
                #     backward_penalty = -0.5 # 倒退惩罚：倒退距离 * 5
                # # if delta_x >= 0.5:
                # #     backward_penalty = 0.3  # 未倒退，无惩罚
                # else:
                #     backward_penalty = delta_x/4  # 未倒退，无惩罚
                vehicle.last_x_position = vehicle.position[0]  # 更新上一步位置

            #5.完成汇入任务奖励   6.完成步数奖励
            merge_success = 0
            step_success = 0
            if vehicle.position[0] > 440 and (vehicle.lane_index in [("b", "c", 0), ("c", "d", 0)]) :
                merge_success += 25
                if self.steps < 35:
                    step_success += 10

            #7.高速奖励
            speed_reward = 0
            speed_reward += 1.5 * np.clip(scaled_speed, 0, 2)

            #8.碰撞奖励
            collision_reward = 0
            collision_reward += 15 * (-1 * vehicle.crashed)

            #9.偏离车道
            # 获取当前车辆所在车道
            Lane_deviation = 0
            lane = self.road.network.get_lane(self.vehicle.lane_index)
            # local_coordinates 返回 (longitudinal, lateral)
            _, lat = lane.local_coordinates(self.vehicle.position)

            # 10.对横向偏移做一个惩罚，让车辆更倾向于居中
            Lane_deviation += -4 * abs(lat)
            print("Lane_deviation:",Lane_deviation)

            #11.进入cd0车道
            merging_lane_cost = 0
            if vehicle.lane_index == ("b", "c", 0):
                merging_lane_cost += 0.3
            if vehicle.lane_index == ("c", "d", 0):
                merging_lane_cost +=  2 * (vehicle.position[0] / cd0_end)  # 进入 cd0 并靠近末端

            print(" 车辆位置是：vehicle.position[0]",vehicle.position[0])

            # 打印每个奖励分量
            print("------------------------------------------主车道行驶（核心任务：车道保持到终点）-------------------------------------------")
            print(f"Reward breakdown at step {self.steps}: "
                  f"alignment_bonus={alignment_bonus:.2f}, "
                  f"steering_penalty={steering_penalty:.2f}, "
                  f"headway_penalty={headway_penalty:.2f}, "
                  f"backward_penalty={backward_penalty:.2f}, "
                  f"merge_success={merge_success:.2f}, "
                  f"step_success={step_success:.2f}, "
                  f"collision_reward={collision_reward:.2f}, "
                  f"merging_lane_cost={merging_lane_cost:.2f}, "
                  f"speed_reward={speed_reward:.2f}, "
                  f"Lane_deviation={Lane_deviation:.2f}, "
                  f"reward_backward={reward_backward:.2f}, ")
            reward = 0
            reward += headway_penalty  + merge_success + collision_reward+ Lane_deviation + reward_backward
            print("reward:", reward)
        else:
            alignment_bonus = 0   #1.位置对齐奖励
            steering_penalty = 0   #2.角度惩罚
            steering_penalty = -3 * max(0, abs(action[1]) - 0.25)
            headway_penalty = 0     # 3.车头时距惩罚

            #4.碰撞奖励
            collision_reward = 0
            collision_reward = 15 * (-1 * vehicle.crashed)

            #5.高速奖励
            speed_reward = 0
            speed_reward = 1.5 * np.clip(scaled_speed, 0, 3)

            #6.汇入前后车距离
            headway_distance = self._compute_headway_distance(vehicle)
            if vehicle.lane_index in [("b", "c", 1), ("b", "c", 0),("c", "d", 0)] and vehicle.speed > 0:
                min_safe_distance = vehicle.speed * 1  # 1 秒车头时距
                max_safe_distance = vehicle.speed * 2  # 3 秒车头时距
                print("headway_distance：", headway_distance)
                Headway_cost = np.log(headway_distance / (1 * vehicle.speed))
                print("Headway_cost:",Headway_cost)
                if headway_distance < min_safe_distance:
                    headway_penalty = (headway_distance - min_safe_distance) *0.2
                if headway_distance > max_safe_distance:
                    headway_penalty = (max_safe_distance - headway_distance) *0.2
                if headway_distance >= min_safe_distance and headway_distance >= max_safe_distance:
                    headway_penalty = (max_safe_distance - min_safe_distance) * 0.15
            else:
                if vehicle.speed==0 :
                    headway_penalty = (10 - headway_distance) * 0.2
                if vehicle.lane_index not in [("b", "c", 1), ("b", "c", 0),("c", "d", 0)]:
                    headway_penalty = -2
                    print("  ")
            headway_cost_reward = headway_penalty

            #7.倒退惩罚
            backward_penalty = 0
            if not hasattr(vehicle, 'last_x_position'):
                vehicle.last_x_position = vehicle.position[0]  # 初次设置上一步位置
            else:
                delta_x = vehicle.position[0] - vehicle.last_x_position  # 计算 x 坐标变化
                if delta_x < 0:
                    backward_penalty = -5 * abs(delta_x)  # 倒退惩罚：倒退距离 * 5
                else:
                    backward_penalty = 0  # 未倒退，无惩罚
                vehicle.last_x_position = vehicle.position[0]  # 更新上一步位置

            # 8.进入目标车道奖励
            # 分阶段奖励
            merging_lane_cost = 0
            if vehicle.lane_index == ("b", "c", 0):
                merging_lane_cost += 0  # 进入 bc1

            print("------------------------------------------匝道汇入（核心任务：安全汇入）-------------------------------------------")
            print(f"Reward breakdown at step {self.steps}: "
                  f"steering_penalty={steering_penalty:.2f}, "
                  f"headway_cost_reward ={headway_cost_reward :.2f}, "
                  f"backward_penalty={backward_penalty:.2f}, "
                  f"collision_reward={collision_reward:.2f}, "
                  f"speed_reward={speed_reward:.2f}, ")
            reward = 0
            reward += steering_penalty + headway_cost_reward + collision_reward + speed_reward + backward_penalty
        # 组合所有奖励分量
        # 汇入时需要的奖励：1.碰撞 2.速度奖励 3.汇入前后车距离  4. 进入目标车道奖励   5.高速奖励
        # 主车道行驶：1.位置对其  2.角度惩罚  3.前车跟着  4.倒退惩罚  5.完成汇入任务奖励  6。总完成步数  7.高速奖励
            print("reward:",reward)
        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        vehicle = self.controlled_vehicles[0]
        lane = self.road.network.get_lane(vehicle.lane_index)

        # 检查车辆是否越界或碰撞
        if not lane.on_lane(vehicle.position):
            info["out_of_bounds"] = True
            done = True
        if vehicle.crashed:
            info["collision"] = True
            done = True
        # if vehicle.lane_index in [("b", "c", 0), ("c", "d", 0),("b", "c", 1),("k", "b", 0)]:
        #     info["out_of_bounds"] = True
        #     done = True
        return obs, reward, done, info

    def _is_terminal(self) -> bool:
        vehicle = self.controlled_vehicles[0]
        if vehicle.position[0] > 440:
            print("vehicle.position[0] > 440")
        return (vehicle.crashed or self.steps > 200 or vehicle.position[0] > 440)

    def _reset(self, num_CAV=0) -> None:
        self._make_road()
        if num_CAV == 0:
            num_CAV = 1
        hdv_ranges = {1: (3, 4), 2: (5, 6), 3: (10, 11), 4: (12, 13)}
        low, high = hdv_ranges.get(self.config["traffic_density"], (3, 4))
        num_HDV = np.random.choice(np.arange(low, high), 1)[0]
        self._make_vehicles(num_CAV, num_HDV)
        self.action_is_safe = True
        self.T = int(self.config["duration"] * self.config["policy_frequency"])
        vehicle = self.controlled_vehicles[0]
        vehicle.has_entered_bc0 = False
        vehicle.steps_on_bc0 = 0
        vehicle.has_calculated_merging_cost = False
        vehicle.merging_cost_value = 0
        vehicle.last_x_position = vehicle.position[0]  # 初始化上一步 x 位置

    def _make_road(self) -> None:
        net = RoadNetwork()
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        net.add_lane("a", "b", StraightLane([0, 0], [sum(self.ends[:2]), 0], line_types=[c, c]))
        net.add_lane("b", "c", StraightLane([sum(self.ends[:2]), 0], [sum(self.ends[:3]), 0], line_types=[c, s]))
        net.add_lane("c", "d", StraightLane([sum(self.ends[:3]), 0], [sum(self.ends), 0], line_types=[c, c]))
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4], [self.ends[0], 6.5 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(self.ends[0], -amplitude), ljk.position(sum(self.ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2 * self.ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(self.ends[1], 0), lkb.position(self.ends[1], 0) + [self.ends[2], 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(self.ends[2], 0)))
        self.road = road

    # def _make_vehicles(self, num_CAV=1, num_HDV=10) -> None:
    #     print("--------------------_make_vehicles-------------------------")
    #     road = self.road
    #     other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
    #     self.controlled_vehicles = []
    #     # spawn_points_s = [100,120, 140, 160, 180, 200, 220, 240,260, 287, 295,310, 335, 370, 410]
    #     spawn_points_s = np.arange(270, 400, 9).tolist()
    #     spawn_points_m = [120, 155, 176, 197, 210, 230, 250, 280, 295, 320, 400]
    #     cav_spawn_points = np.linspace(302, 304, num_CAV, endpoint=True)
    #     available_s_points = spawn_points_s.copy()
    #     spawn_point_s_h = np.random.choice(available_s_points, num_HDV, replace=False)
    #     spawn_point_s_h = list(spawn_point_s_h)
    #     initial_speed = np.random.rand(num_CAV + num_HDV) * 2 + 15
    #     loc_noise = np.random.rand(num_CAV + num_HDV) * 1 - 0.5
    #     initial_speed = list(initial_speed)
    #     loc_noise = list(loc_noise)
    #     jk0_lane = road.network.get_lane(("j", "k", 0))
    #     if isinstance(jk0_lane, StraightLane):
    #         y_center = jk0_lane.start[1]
    #     else:
    #         y_center = jk0_lane.position(cav_spawn_points[0], 0)[1]
    #     for i in range(num_CAV):
    #         x_pos = cav_spawn_points[i] + loc_noise.pop(0)
    #         y_offset = np.random.uniform(-5, -2)
    #         if isinstance(jk0_lane, StraightLane):
    #             #y_pos = y_center + y_offset
    #             y_pos = y_center - 4.5
    #         else:
    #             y_center = jk0_lane.position(x_pos, 0)[1]
    #             #y_pos = y_center + y_offset
    #             y_pos = y_center - 4.5
    #         ego_vehicle = self.action_type.vehicle_class(road, [x_pos, y_pos], speed=initial_speed.pop(0))
    #         self.controlled_vehicles.append(ego_vehicle)
    #         road.vehicles.append(ego_vehicle)
    #         #新补充CAV车辆角度设置
    #
    #     for _ in range(num_HDV // 2):
    #         road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(
    #             spawn_point_s_h.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0)))
    #     for _ in range(num_HDV - num_HDV // 2):
    #         road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("b", "c", 0)).position(
    #             spawn_point_s_h.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0)))

    # def _make_vehicles(self, num_CAV=1, num_HDV=10) -> None:
    #     print("--------------------_make_vehicles (fixed CAV spawn)-------------------------")
    #     road = self.road
    #     other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
    #     self.controlled_vehicles = []
    #
    #     # ===== 读固定出生配置 =====
    #     fixed_spawn = bool(self.config.get("cav_fixed_spawn", True))
    #     cav_x = float(self.config.get("cav_spawn_x", 303.0))
    #     cav_y_offset = float(self.config.get("cav_y_offset", -4.5))
    #     cav_speed = float(self.config.get("cav_speed", 16.0))
    #     cav_heading = self.config.get("cav_heading", None)  # None or float (rad)
    #
    #     # ===== 车道与几何 =====
    #     jk0_lane = road.network.get_lane(("j", "k", 0))
    #     if isinstance(jk0_lane, StraightLane):
    #         # 直线段：y 中心即起点的 y
    #         y_center = jk0_lane.start[1]
    #     else:
    #         # 正弦段：取给定 x 处的中心线 y
    #         y_center = jk0_lane.position(cav_x, 0)[1]
    #
    #     # ======= 生成 CAV（受控车） =======
    #     # 注意：如果 num_CAV > 1，就在 cav_x 附近均匀铺开（仍然是“确定性”）
    #     if num_CAV >= 1:
    #         if num_CAV == 1:
    #             xs = [cav_x]
    #         else:
    #             # 多车时：以 cav_x 为中心做一个很小的范围均匀分布（确定性）
    #             xs = list(np.linspace(cav_x - 0.5, cav_x + 0.5, num_CAV, endpoint=True))
    #         for i, x_pos in enumerate(xs):
    #             y_pos = y_center + cav_y_offset
    #             ego_vehicle = self.action_type.vehicle_class(road, [float(x_pos), float(y_pos)], speed=cav_speed)
    #             # 可选：固定朝向
    #             if cav_heading is not None:
    #                 try:
    #                     ego_vehicle.heading = float(cav_heading)
    #                 except Exception:
    #                     pass
    #             self.controlled_vehicles.append(ego_vehicle)
    #             road.vehicles.append(ego_vehicle)
    #
    #     # ======= 生成 HDV（与原来基本一致） =======
    #     # 主线车辆的出生位置池
    #     spawn_points_s = np.arange(270, 400, 9).tolist()
    #     available_s_points = spawn_points_s.copy()
    #     spawn_point_s_h = np.random.choice(available_s_points, num_HDV, replace=False).tolist()
    #
    #     # HDV 的速度与横向噪声（只给 HDV 用，避免影响 CAV）
    #     initial_speed_hdv = (np.random.rand(num_HDV) * 2 + 15).tolist()
    #     loc_noise_hdv = (np.random.rand(num_HDV) * 1 - 0.5).tolist()
    #
    #     # 一半在 a->b，一半在 b->c
    #     half = num_HDV // 2
    #     for _ in range(half):
    #         road.vehicles.append(
    #             other_vehicles_type(
    #                 road,
    #                 road.network.get_lane(("a", "b", 0)).position(
    #                     float(spawn_point_s_h.pop(0) + loc_noise_hdv.pop(0)), 0
    #                 ),
    #                 speed=float(initial_speed_hdv.pop(0))
    #             )
    #         )
    #     for _ in range(num_HDV - half):
    #         road.vehicles.append(
    #             other_vehicles_type(
    #                 road,
    #                 road.network.get_lane(("b", "c", 0)).position(
    #                     float(spawn_point_s_h.pop(0) + loc_noise_hdv.pop(0)), 0
    #                 ),
    #                 speed=float(initial_speed_hdv.pop(0))
    #             )
    #         )

    def _make_vehicles(self, num_CAV=1, num_HDV=10) -> None:
        print("--------------------_make_vehicles (fixed CAV spawn)-------------------------")
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []

        # ===== 读固定出生配置 =====
        fixed_spawn = bool(self.config.get("cav_fixed_spawn", True))
        cav_x = float(self.config.get("cav_spawn_x", 307.0))
        cav_y_offset = float(self.config.get("cav_y_offset", -4.2))
        cav_speed = float(self.config.get("cav_speed", 16.0))
        cav_heading = self.config.get("cav_heading", None)  # None or float (rad)

        # ===== 车道与几何 =====
        jk0_lane = road.network.get_lane(("j", "k", 0))
        if isinstance(jk0_lane, StraightLane):
            # 直线段：y 中心即起点的 y
            y_center = jk0_lane.start[1]
        else:
            # 正弦段：取给定 x 处的中心线 y
            y_center = jk0_lane.position(cav_x, 0)[1]

        # ======= 生成 CAV（受控车） =======
        # 注意：如果 num_CAV > 1，就在 cav_x 附近均匀铺开（仍然是“确定性”）
        if num_CAV >= 1:
            if num_CAV == 1:
            #     xs = [cav_x]
            # else:
            #     # 多车时：以 cav_x 为中心做一个很小的范围均匀分布（确定性）
            #     xs = list(np.linspace(cav_x - 0.5, cav_x + 0.5, num_CAV, endpoint=True))
                xs = [cav_x]
                y_pos = 5.68
            for i, x_pos in enumerate(xs):
                #y_pos = y_center + cav_y_offset
                #y_pos = y_center
                y_pos = 5.18
                x_pos = 307.13
                ego_vehicle = self.action_type.vehicle_class(road, [float(x_pos), float(y_pos)], speed=cav_speed)
                # 可选：固定朝向
                if cav_heading is not None:
                    try:
                        ego_vehicle.heading = float(cav_heading)
                    except Exception:
                        pass
                self.controlled_vehicles.append(ego_vehicle)
                road.vehicles.append(ego_vehicle)

        # ======= 生成 HDV（与原来基本一致） =======
        # 主线车辆的出生位置池
        spawn_points_s = np.arange(270, 400, 9).tolist()
        available_s_points = spawn_points_s.copy()
        spawn_point_s_h = np.random.choice(available_s_points, num_HDV, replace=False).tolist()

        # HDV 的速度与横向噪声（只给 HDV 用，避免影响 CAV）
        initial_speed_hdv = (np.random.rand(num_HDV) * 2 + 15).tolist()
        loc_noise_hdv = (np.random.rand(num_HDV) * 1 - 0.5).tolist()

        # 一半在 a->b，一半在 b->c
        half = num_HDV // 2
        for _ in range(half):
            road.vehicles.append(
                other_vehicles_type(
                    road,
                    road.network.get_lane(("a", "b", 0)).position(
                        float(spawn_point_s_h.pop(0) + loc_noise_hdv.pop(0)), 0
                    ),
                    speed=float(initial_speed_hdv.pop(0))
                )
            )
        for _ in range(num_HDV - half):
            road.vehicles.append(
                other_vehicles_type(
                    road,
                    road.network.get_lane(("b", "c", 0)).position(
                        float(spawn_point_s_h.pop(0) + loc_noise_hdv.pop(0)), 0
                    ),
                    speed=float(initial_speed_hdv.pop(0))
                )
            )

    def terminate(self):
        return

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds


register(id='merge-v33', entry_point='highway_env.envs:MergeEnv')
