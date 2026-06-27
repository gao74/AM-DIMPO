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
        return self._agent_reward(action, self.controlled_vehicles[0])

    def _compute_headway_distance(self, vehicle):
        positions = np.array([v.position for v in self.road.vehicles])
        kdtree = KDTree(positions)
        neighbors = kdtree.query_ball_point(vehicle.position, 80)
        min_distance = 150
        target_lanes = []
        if vehicle.lane_index == ("b", "c", 1):
            target_lanes = [("b", "c", 0), ("c", "d", 0)]

        for idx in neighbors:
            v = self.road.vehicles[idx]
            if v is not vehicle:
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
        cd0_end = sum(self.ends)
        scaled_speed = utils.lmap(vehicle.speed, [5, 20], [0, 1])

        def calculate_backward_distance(vehicle, target_lanes=None):
            backward_distance = 300
            for v in self.road.vehicles:
                if not hasattr(v, 'lane_index'):
                    continue
                if v is not vehicle:
                    in_target_lanes = target_lanes and v.lane_index in target_lanes
                    same_lane = not target_lanes and v.lane_index == vehicle.lane_index
                    if in_target_lanes or same_lane:
                        if v.position[0] < vehicle.position[0]:
                            distance = vehicle.position[0] - v.position[0]
                            if distance < backward_distance:
                                backward_distance = distance
            return backward_distance if backward_distance < 300 else None

        headway_distance = self._compute_headway_distance(vehicle)
        if vehicle.lane_index in [("b", "c", 0), ("c", "d", 0)] and vehicle.speed > 0:
            Headway_cost = np.log(headway_distance / (1 * vehicle.speed))
        else:
            Headway_cost = 0
        safety_bonus = 1 if Headway_cost > 0 else 0
        headway_cost_reward = 2 + 2 * (Headway_cost if Headway_cost < 0 else 0)

        reward_backward = 0
        has_not_forward_car = False
        if vehicle.lane_index in [("b", "c", 0), ("c", "d", 0)] and vehicle.speed > 0:
            if headway_distance > 100:
                has_not_forward_car = True
                Headway_cost = 0
                backward_distance = calculate_backward_distance(vehicle, [("b", "c", 0), ("c", "d", 0)])
                reward_backward = backward_distance / 20 - 1 if backward_distance is not None else 0

        alignment_bonus = 0
        if vehicle.lane_index in [("b", "c", 0), ("c", "d", 0)]:
            alignment_bonus += 0.8
            steering_penalty = -4 * max(0, abs(action[1]) - 0.25)
            if vehicle.heading == 0:
                steering_penalty += 0.8
            if -0.2 <= vehicle.heading <= 0.2:
                steering_penalty += 0.5

            headway_penalty = 0
            if vehicle.lane_index in [("b", "c", 0), ("c", "d", 0)] and vehicle.speed > 0 and not has_not_forward_car:
                min_safe_distance = vehicle.speed * 1
                max_safe_distance = vehicle.speed * 1.75
                if headway_distance < min_safe_distance:
                    headway_penalty = (headway_distance - min_safe_distance) * 0.2
                elif headway_distance > max_safe_distance:
                    headway_penalty = (max_safe_distance - headway_distance) * 0.2
                else:
                    headway_penalty = (max_safe_distance - min_safe_distance) * 0.2
            else:
                headway_penalty = 0

            backward_penalty = 0
            if not hasattr(vehicle, 'last_x_position'):
                vehicle.last_x_position = vehicle.position[0]
            else:
                delta_x = vehicle.position[0] - vehicle.last_x_position
                backward_penalty = 5 * delta_x - 1
                vehicle.last_x_position = vehicle.position[0]

            merge_success = 0
            step_success = 0
            if vehicle.position[0] > 440 and vehicle.lane_index in [("b", "c", 0), ("c", "d", 0)]:
                merge_success += 25
                if self.steps < 35:
                    step_success += 10

            speed_reward = 1.5 * np.clip(scaled_speed, 0, 2)
            collision_reward = 15 * (-1 * vehicle.crashed)

            lane = self.road.network.get_lane(self.vehicle.lane_index)
            _, lat = lane.local_coordinates(self.vehicle.position)
            Lane_deviation = -4 * abs(lat)

            merging_lane_cost = 0
            if vehicle.lane_index == ("b", "c", 0):
                merging_lane_cost += 0.3
            if vehicle.lane_index == ("c", "d", 0):
                merging_lane_cost += 2 * (vehicle.position[0] / cd0_end)

            reward = headway_penalty + merge_success + collision_reward + Lane_deviation + reward_backward
        else:
            steering_penalty = -3 * max(0, abs(action[1]) - 0.25)
            headway_penalty = 0
            collision_reward = 15 * (-1 * vehicle.crashed)
            speed_reward = 1.5 * np.clip(scaled_speed, 0, 3)

            if vehicle.lane_index in [("b", "c", 1), ("b", "c", 0), ("c", "d", 0)] and vehicle.speed > 0:
                min_safe_distance = vehicle.speed * 1
                max_safe_distance = vehicle.speed * 2
                if headway_distance < min_safe_distance:
                    headway_penalty = (headway_distance - min_safe_distance) * 0.2
                elif headway_distance > max_safe_distance:
                    headway_penalty = (max_safe_distance - headway_distance) * 0.2
                else:
                    headway_penalty = (max_safe_distance - min_safe_distance) * 0.15
            else:
                headway_penalty = -2 if vehicle.speed != 0 else (10 - headway_distance) * 0.2

            backward_penalty = 0
            if not hasattr(vehicle, 'last_x_position'):
                vehicle.last_x_position = vehicle.position[0]
            else:
                delta_x = vehicle.position[0] - vehicle.last_x_position
                backward_penalty = -5 * abs(delta_x) if delta_x < 0 else 0
                vehicle.last_x_position = vehicle.position[0]

            merging_lane_cost = 0
            reward = steering_penalty + headway_penalty + collision_reward + speed_reward + backward_penalty
        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        vehicle = self.controlled_vehicles[0]
        lane = self.road.network.get_lane(vehicle.lane_index)

        if not lane.on_lane(vehicle.position):
            info["out_of_bounds"] = True
            done = True
        if vehicle.crashed:
            info["collision"] = True
            done = True
        return obs, reward, done, info

    def _is_terminal(self) -> bool:
        vehicle = self.controlled_vehicles[0]
        return vehicle.crashed or self.steps > 200 or vehicle.position[0] > 440

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
        vehicle.last_x_position = vehicle.position[0]

    def _make_road(self) -> None:
        net = RoadNetwork()
        self.ends = [150, 80, 80, 150]
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]

        # 双主车道
        for i in range(2):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(self.ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("b", "c", StraightLane([sum(self.ends[:2]), y[i]], [sum(self.ends[:3]), y[i]], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([sum(self.ends[:3]), y[i]], [sum(self.ends), y[i]], line_types=line_type[i]))

        # 汇入匝道
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4], [self.ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
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

    def _make_vehicles(self, num_CAV=1, num_HDV=10) -> None:
        print("--------------------AV 在 KB 匝道车道中心生成，HDV 平均分配到两个主车道-------------------------")
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []

        # ===================== 修复：AV 生成在 KB 匝道车道 (k,b,0) 中心位置 =====================
        # 目标车道：KB 匝道车道（你要求的车道）
        ramp_lane = road.network.get_lane(("k", "b", 0))
        # 纵向位置：100m处 | 横向偏移：0.5（向右微调，居中，彻底解决靠左问题）
        # position(longitudinal, lateral)：lateral>0 向右，=0 中心，<0 向左
        av_position = ramp_lane.position(40, 0.5)
        # 生成AV车辆
        ego_vehicle = self.action_type.vehicle_class(road, av_position, speed=25)
        self.controlled_vehicles.append(ego_vehicle)
        road.vehicles.append(ego_vehicle)

        # HDV 平均分配到 主车道0 和 主车道1（不变）
        spawn_points = np.arange(100, 350, 12).tolist()
        np.random.shuffle(spawn_points)
        selected = np.random.choice(spawn_points, num_HDV, replace=False).tolist()
        speeds = (np.random.rand(num_HDV) * 3 + 20).tolist()

        half = num_HDV // 2
        # 一半 HDV 在主车道 0：a→b→c→d 车道0
        for _ in range(half):
            pos = selected.pop(0)
            v = other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(pos, 0), speed=speeds.pop(0))
            road.vehicles.append(v)

        # 一半 HDV 在主车道 1：a→b→c→d 车道1
        for _ in range(num_HDV - half):
            pos = selected.pop(0)
            v = other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(pos, 0), speed=speeds.pop(0))
            road.vehicles.append(v)

    def terminate(self):
        return

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds


register(id='merge-v50', entry_point='highway_env.envs:MergeEnv')