# merge_env_v41_persistent_2cav_shift_hdv_ratio_0p2.py
# 修改：第二个CAV触发点比例固定为 0.2（20% 路程）
# 不依赖 gym.make(..., config=...)，直接写死在 default_config 里
# 其它逻辑不变：HDV 左移100、最多2车管线、reward原样、bc1逻辑、persistent traffic、continuous respawn

import numpy as np
from gym.envs.registration import register
from typing import Tuple
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.objects import Obstacle
from highway_env.vehicle.kinematics import Vehicle
from scipy.spatial import KDTree


class MergeEnv(AbstractEnv):
    ends = [200, 100, 100, 100]  # jk, kb, bc, cd；总长=500

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {"type": "Kinematics", "vehicles_count": 10},
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
                "steering_range": [-1, 1]
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
            "render_agent_indicators": True,

            # persistent traffic + bc1
            "persistent_traffic": True,
            "continuous_respawn_cav": True,
            "hdv_target_count": 10,
            "recycle_hdv": True,

            "hdv_turn_bc1_prob": 0.10,
            "bc1_trigger_x0": None,
            "bc1_trigger_x1": None,
            "bc1_prob_power": 2.0,
            "bc1_return_margin": 18.0,
            "bc1_return_lane": ("b", "c", 0),

            # CAV spawn + 2-CAV pipeline
            "cav_spawn_x": 307.0,
            "cav_spawn_speed": 16.0,
            "cav_goal_x": 440.0,
            "cav_spawn_lane_candidates": [("b", "c", 0), ("k", "b", 0), ("j", "k", 0)],

            # ✅ 固定比例=0.2（不走外部 config 注入）
            "cav_spawn_ratio": 0.7,

            # HDV spawn shift left by 100
            "hdv_spawn_shift_left": 25.0,
        })
        return config

    # ----------------------------
    # Spawn helpers
    # ----------------------------
    def _lane_length(self, lane) -> float:
        L = getattr(lane, "length", 0.0)
        return float(L() if callable(L) else L)

    def _best_spawn_on_lanes_by_x(self, x_desired: float):
        candidates = self.config.get("cav_spawn_lane_candidates", [("b", "c", 0), ("k", "b", 0), ("j", "k", 0)])
        best = None  # (abs_err, pos, heading)

        for lane_index in candidates:
            try:
                lane = self.road.network.get_lane(lane_index)
            except Exception:
                continue

            L = self._lane_length(lane)
            if not np.isfinite(L) or L <= 1e-6:
                continue

            grid = np.linspace(0.0, L, num=1001)
            xs = np.empty_like(grid)
            for i, s in enumerate(grid):
                xs[i] = lane.position(float(s), 0.0)[0]

            idx = int(np.argmin(np.abs(xs - x_desired)))
            s_best = float(grid[idx])
            pos = lane.position(s_best, 0.0)
            heading = float(lane.heading_at(s_best)) if hasattr(lane, "heading_at") else 0.0

            err = float(abs(pos[0] - x_desired))
            if (best is None) or (err < best[0]):
                best = (err, [float(pos[0]), float(pos[1])], heading)

        if best is None:
            lane = self.road.network.get_lane(candidates[0])
            pos = lane.position(0.0, 0.0)
            heading = float(lane.heading_at(0.0)) if hasattr(lane, "heading_at") else 0.0
            return [float(pos[0]), float(pos[1])], heading

        return best[1], best[2]

    def _set_cav_spawn_cache(self):
        """
        缓存 CAV 原始生成点（pos/heading），并计算触发点 x
        触发点 = x_spawn_actual + ratio * (goal_x - x_spawn_actual)
        ratio 固定为 0.2（从 config 读，但 config 内已写死 0.2）
        """
        x_desired = float(self.config["cav_spawn_x"])
        self._cav_spawn_pos, self._cav_spawn_heading = self._best_spawn_on_lanes_by_x(x_desired)
        self._cav_spawn_x_actual = float(self._cav_spawn_pos[0])

        goal_x = float(self.config.get("cav_goal_x", 440.0))
        ratio = float(self.config.get("cav_spawn_ratio", 0.2))  # ✅ 这里默认 0.2
        ratio = float(np.clip(ratio, 0.0, 1.0))

        self._cav_trigger_x = self._cav_spawn_x_actual + ratio * (goal_x - self._cav_spawn_x_actual)

    # ----------------------------
    # HDV bc1 logic helpers
    # ----------------------------
    def _init_turn_window(self):
        if self.config.get("bc1_trigger_x0", None) is None or self.config.get("bc1_trigger_x1", None) is None:
            bc_start = float(sum(self.ends[:2]))
            bc_end = float(sum(self.ends[:3]))
            self.config["bc1_trigger_x0"] = bc_start + 10.0
            self.config["bc1_trigger_x1"] = bc_end - 25.0

    def _tag_hdv_once(self, v: Vehicle):
        if getattr(v, "_is_controlled", False):
            return
        if getattr(v, "_is_queued_cav", False):
            return
        if not hasattr(v, "_bc1_decided"):
            v._bc1_decided = False
            v._turn_to_bc1 = False
            v._bc1_state = 0
            v._crash_at_bc1_end = False

    def _turn_probability_curve(self, x: float) -> float:
        x0 = float(self.config["bc1_trigger_x0"])
        x1 = float(self.config["bc1_trigger_x1"])
        pmax = float(self.config["hdv_turn_bc1_prob"])
        if x <= x0:
            return 0.0
        if x >= x1:
            return pmax
        t = (x - x0) / max(x1 - x0, 1e-6)
        power = float(self.config.get("bc1_prob_power", 2.0))
        return float(np.clip(pmax * (t ** power), 0.0, pmax))

    def _update_hdv_bc1_logic(self):
        bc1 = ("b", "c", 1)
        bc0 = ("b", "c", 0)
        ret_lane = tuple(self.config.get("bc1_return_lane", ("b", "c", 0)))
        return_margin = float(self.config.get("bc1_return_margin", 18.0))
        bc1_end_x = float(sum(self.ends[:2]) + self.ends[2])

        for v in self.road.vehicles:
            if v in self.controlled_vehicles:
                continue
            if getattr(v, "_is_queued_cav", False):
                continue
            if not hasattr(v, "lane_index") or not hasattr(v, "position"):
                continue

            self._tag_hdv_once(v)

            if (not getattr(v, "_bc1_decided", False)) and (v.lane_index == bc0):
                x = float(v.position[0])
                x0 = float(self.config["bc1_trigger_x0"])
                x1 = float(self.config["bc1_trigger_x1"])
                if x0 <= x <= x1:
                    p = self._turn_probability_curve(x)
                    v._turn_to_bc1 = (np.random.rand() < p)
                    v._bc1_decided = True
                    if v._turn_to_bc1 and hasattr(v, "target_lane_index"):
                        v.target_lane_index = bc1
                        v._bc1_state = 1

            if getattr(v, "_turn_to_bc1", False) and (v.lane_index == bc1):
                v._bc1_state = 2
                if float(v.position[0]) >= (bc1_end_x - return_margin):
                    if hasattr(v, "target_lane_index"):
                        v.target_lane_index = ret_lane
                        v._bc1_state = 3

    # ----------------------------
    # 2-CAV pipeline
    # ----------------------------
    def _spawn_ego_at_spawn(self):
        road = self.road
        self._set_cav_spawn_cache()
        cav_speed = float(self.config["cav_spawn_speed"])

        ego = self.action_type.vehicle_class(
            road,
            list(self._cav_spawn_pos),
            heading=float(self._cav_spawn_heading),
            speed=float(cav_speed)
        )
        ego._is_controlled = True
        self.controlled_vehicles = [ego]
        road.vehicles.append(ego)

        ego.has_entered_bc0 = False
        ego.steps_on_bc0 = 0
        ego.has_calculated_merging_cost = False
        ego.merging_cost_value = 0
        ego.last_x_position = ego.position[0]

        ego._spawned_next_cav = False

    def _spawn_queued_cav_at_spawn(self):
        if getattr(self, "_queued_cav", None) is not None:
            return

        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        cav_speed = float(self.config["cav_spawn_speed"])

        v = other_vehicles_type(
            road,
            list(self._cav_spawn_pos),
            heading=float(self._cav_spawn_heading),
            speed=float(cav_speed)
        )
        v._is_queued_cav = True
        v._is_controlled = False
        road.vehicles.append(v)
        self._queued_cav = v

    def _remove_vehicle(self, v):
        if v is None:
            return
        try:
            if v in self.road.vehicles:
                self.road.vehicles.remove(v)
        except Exception:
            pass

    def _promote_queued_to_ego(self):
        if getattr(self, "_queued_cav", None) is None:
            return False

        q = self._queued_cav
        q_pos = [float(q.position[0]), float(q.position[1])]
        q_speed = float(getattr(q, "speed", self.config["cav_spawn_speed"]))
        q_heading = float(getattr(q, "heading", 0.0))

        self._remove_vehicle(q)
        self._queued_cav = None

        if self.controlled_vehicles:
            self._remove_vehicle(self.controlled_vehicles[0])
        self.controlled_vehicles = []

        ego = self.action_type.vehicle_class(
            self.road,
            q_pos,
            heading=q_heading,
            speed=q_speed
        )
        ego._is_controlled = True
        self.controlled_vehicles = [ego]
        self.road.vehicles.append(ego)

        ego.has_entered_bc0 = False
        ego.steps_on_bc0 = 0
        ego.has_calculated_merging_cost = False
        ego.merging_cost_value = 0
        ego.last_x_position = ego.position[0]
        ego._spawned_next_cav = False

        return True

    def _cleanup_queued_if_needed(self):
        q = getattr(self, "_queued_cav", None)
        if q is None:
            return
        end_x = float(sum(self.ends))
        try:
            lane = self.road.network.get_lane(q.lane_index)
            on_lane = lane.on_lane(q.position)
        except Exception:
            on_lane = False
        if (getattr(q, "crashed", False)
                or (not on_lane)
                or (float(q.position[0]) > end_x + 5.0)):
            self._remove_vehicle(q)
            self._queued_cav = None

    # ----------------------------
    # Replay traffic recycle (只管 HDV)
    # ----------------------------
    def _spawn_one_hdv(self):
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        shift = float(self.config.get("hdv_spawn_shift_left", 0.0))

        if np.random.rand() < 0.5:
            lane = road.network.get_lane(("a", "b", 0))
            world_x = float(np.random.uniform(30, 180) - shift)
            world_x = max(world_x, 0.0)
        else:
            lane = road.network.get_lane(("b", "c", 0))
            bc_start_x = float(sum(self.ends[:2]))
            world_x = float(np.random.uniform(bc_start_x + 10, bc_start_x + 80) - shift)

        start_x = float(getattr(lane, "start", [0.0, 0.0])[0])
        s_local = world_x - start_x

        x, y = lane.position(float(s_local), 0.0)
        speed = float(np.random.rand() * 2 + 15)
        v = other_vehicles_type(road, [float(x), float(y)], speed=speed)
        v._is_controlled = False
        self._tag_hdv_once(v)
        road.vehicles.append(v)

    def _recycle_hdv_if_needed(self):
        if not self.config.get("recycle_hdv", True):
            return

        end_x = float(sum(self.ends))
        bc1 = ("b", "c", 1)
        bc1_end_x = float(sum(self.ends[:2]) + self.ends[2])

        keep = []
        removed = 0

        for v in self.road.vehicles:
            if v in self.controlled_vehicles:
                keep.append(v)
                continue
            if getattr(v, "_is_queued_cav", False):
                keep.append(v)
                continue
            if not hasattr(v, "lane_index") or not hasattr(v, "position"):
                keep.append(v)
                continue

            self._tag_hdv_once(v)

            if getattr(v, "crashed", False):
                if (v.lane_index == bc1) and (float(v.position[0]) >= bc1_end_x - 2.0):
                    v._crash_at_bc1_end = True
                removed += 1
                continue

            if float(v.position[0]) > end_x + 5.0:
                removed += 1
                continue

            try:
                lane = self.road.network.get_lane(v.lane_index)
                if not lane.on_lane(v.position):
                    removed += 1
                    continue
            except Exception:
                removed += 1
                continue

            keep.append(v)

        self.road.vehicles = keep

        for _ in range(removed):
            self._spawn_one_hdv()

        target = int(self.config.get("hdv_target_count", 10))
        current_hdv = sum(
            1 for v in self.road.vehicles
            if (v not in self.controlled_vehicles)
            and (not getattr(v, "_is_queued_cav", False))
            and hasattr(v, "lane_index")
        )
        while current_hdv < target:
            self._spawn_one_hdv()
            current_hdv += 1

    # ----------------------------
    # ✅ reward：100% 原样保留（同你粘贴 reward）
    # ----------------------------
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

    # ===== reward 原样 =====
    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        cd0_end = sum(self.ends)
        scaled_speed = utils.lmap(vehicle.speed, [5, 20], [0, 1])
        print(f"Current lane: {vehicle.lane_index}")

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
            if headway_distance > 100:
                print("无前车，有后车")
                has_not_forward_car = True
                Headway_cost = 0
                backward_distance = calculate_backward_distance(vehicle, [("b", "c", 0), ("c", "d", 0)])
                if backward_distance is not None:
                    reward_backward = backward_distance / 20 - 1
                    print("后车距离奖励：",reward_backward)
                else:
                    reward_backward = 0
            Headway_cost = np.log(headway_distance / (1 * vehicle.speed))
            print("np.log(headway_distance / (1 * vehicle.speed))：", np.log(headway_distance / (1 * vehicle.speed)))

        alignment_bonus = 0

        if (vehicle.lane_index == ("b", "c", 0) or vehicle.lane_index == ("c", "d", 0)):
            reward = 0
            alignment_bonus += 0.8

            steering_penalty = 0
            steering_penalty = -4 * max(0, abs(action[1]) - 0.25)
            if vehicle.heading == 0:
                steering_penalty += 0.8
            if vehicle.heading <= 0.2 and vehicle.heading>=-0.2:
                steering_penalty += 0.5

            headway_penalty = 0
            if vehicle.lane_index in [("b", "c", 0), ("c", "d", 0)] and vehicle.speed > 0 and has_not_forward_car==False:
                print("vehicle.speed:",vehicle.speed)
                min_safe_distance = vehicle.speed * 1
                max_safe_distance = vehicle.speed * 1.75
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

            backward_penalty = 0
            if not hasattr(vehicle, 'last_x_position'):
                vehicle.last_x_position = vehicle.position[0]
            else:
                delta_x = vehicle.position[0] - vehicle.last_x_position
                backward_penalty = 5 * delta_x - 1
                vehicle.last_x_position = vehicle.position[0]

            merge_success = 0
            step_success = 0
            if vehicle.position[0] > 440 and (vehicle.lane_index in [("b", "c", 0), ("c", "d", 0)]) :
                merge_success += 25
                if self.steps < 35:
                    step_success += 10

            speed_reward = 0
            speed_reward += 1.5 * np.clip(scaled_speed, 0, 2)

            collision_reward = 0
            collision_reward += 15 * (-1 * vehicle.crashed)

            Lane_deviation = 0
            lane = self.road.network.get_lane(self.vehicle.lane_index)
            _, lat = lane.local_coordinates(self.vehicle.position)
            Lane_deviation += -4 * abs(lat)
            print("Lane_deviation:",Lane_deviation)

            merging_lane_cost = 0
            if vehicle.lane_index == ("b", "c", 0):
                merging_lane_cost += 0.3
            if vehicle.lane_index == ("c", "d", 0):
                merging_lane_cost +=  2 * (vehicle.position[0] / cd0_end)

            print(" 车辆位置是：vehicle.position[0]",vehicle.position[0])

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
            reward += alignment_bonus + steering_penalty + headway_penalty + backward_penalty + merge_success + step_success + speed_reward + collision_reward + merging_lane_cost + Lane_deviation + reward_backward
            print("reward:", reward)
        else:
            alignment_bonus = 0
            steering_penalty = 0
            steering_penalty = -3 * max(0, abs(action[1]) - 0.25)
            headway_penalty = 0

            collision_reward = 0
            collision_reward = 15 * (-1 * vehicle.crashed)

            speed_reward = 0
            speed_reward = 1.5 * np.clip(scaled_speed, 0, 3)

            headway_distance = self._compute_headway_distance(vehicle)
            if vehicle.lane_index in [("b", "c", 1), ("b", "c", 0),("c", "d", 0)] and vehicle.speed > 0:
                min_safe_distance = vehicle.speed * 1
                max_safe_distance = vehicle.speed * 2
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

            backward_penalty = 0
            if not hasattr(vehicle, 'last_x_position'):
                vehicle.last_x_position = vehicle.position[0]
            else:
                delta_x = vehicle.position[0] - vehicle.last_x_position
                if delta_x < 0:
                    backward_penalty = -5 * abs(delta_x)
                else:
                    backward_penalty = 0
                vehicle.last_x_position = vehicle.position[0]

            merging_lane_cost = 0
            if vehicle.lane_index == ("b", "c", 0):
                merging_lane_cost += 0

            print("------------------------------------------匝道汇入（核心任务：安全汇入）-------------------------------------------")
            print(f"Reward breakdown at step {self.steps}: "
                  f"steering_penalty={steering_penalty:.2f}, "
                  f"headway_cost_reward ={headway_cost_reward :.2f}, "
                  f"backward_penalty={backward_penalty:.2f}, "
                  f"collision_reward={collision_reward:.2f}, "
                  f"speed_reward={speed_reward:.2f}, ")
            reward = 0
            reward += steering_penalty + headway_cost_reward + collision_reward + speed_reward + backward_penalty
            print("reward:",reward)
        return reward

    # ----------------------------
    # Core env methods
    # ----------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = super().step(action)

        self._cleanup_queued_if_needed()
        self._update_hdv_bc1_logic()
        self._recycle_hdv_if_needed()

        ego = self.controlled_vehicles[0]

        # ✅ 触发点：ratio=0.2
        if (not getattr(ego, "_spawned_next_cav", False)) and (float(ego.position[0]) >= float(self._cav_trigger_x)):
            self._spawn_queued_cav_at_spawn()
            ego._spawned_next_cav = True

        # ego terminal check
        try:
            lane = self.road.network.get_lane(ego.lane_index)
            on_lane = lane.on_lane(ego.position)
        except Exception:
            on_lane = False

        goal_x = float(self.config.get("cav_goal_x", 440.0))
        cav_terminal = False
        if not on_lane:
            info["out_of_bounds"] = True
            cav_terminal = True
        if ego.crashed:
            info["collision"] = True
            cav_terminal = True
        if float(ego.position[0]) > goal_x:
            info["reach_goal"] = True
            cav_terminal = True

        if self.config.get("continuous_respawn_cav", True) and cav_terminal:
            info["cav_respawned"] = True

            promoted = self._promote_queued_to_ego()
            if not promoted:
                self._remove_vehicle(ego)
                self.controlled_vehicles = []
                self._spawn_ego_at_spawn()

            done = False

        return obs, reward, done, info

    def _is_terminal(self) -> bool:
        if self.config.get("continuous_respawn_cav", True):
            return False
        ego = self.controlled_vehicles[0]
        goal_x = float(self.config.get("cav_goal_x", 440.0))
        return (ego.crashed or self.steps > 200 or float(ego.position[0]) > goal_x)

    def _reset(self, num_CAV=0) -> None:
        if not hasattr(self, "_persistent_initialized"):
            self._persistent_initialized = False

        if self.config.get("persistent_traffic", True) and self._persistent_initialized:
            if getattr(self, "_queued_cav", None) is not None:
                self._remove_vehicle(self._queued_cav)
            self._queued_cav = None

            if self.controlled_vehicles:
                self._remove_vehicle(self.controlled_vehicles[0])
            self.controlled_vehicles = []

            self._spawn_ego_at_spawn()

            self.action_is_safe = True
            self.T = int(self.config["duration"] * self.config["policy_frequency"])
            self.steps = 0
            return

        self._make_road()
        self._init_turn_window()

        self._queued_cav = None
        self._spawn_ego_at_spawn()

        num_HDV = int(self.config.get("hdv_target_count", 10))
        self._make_hdv(num_HDV)

        self.action_is_safe = True
        self.T = int(self.config["duration"] * self.config["policy_frequency"])
        self.steps = 0
        self._persistent_initialized = True

    def _make_road(self) -> None:
        net = RoadNetwork()
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

        net.add_lane("a", "b", StraightLane([0, 0], [sum(self.ends[:2]), 0], line_types=[c, c]))
        net.add_lane("b", "c", StraightLane([sum(self.ends[:2]), 0], [sum(self.ends[:3]), 0], line_types=[c, s]))
        net.add_lane("c", "d", StraightLane([sum(self.ends[:3]), 0], [sum(self.ends), 0], line_types=[c, c]))

        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4], [self.ends[0], 6.5 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(
            ljk.position(self.ends[0], -amplitude),
            ljk.position(sum(self.ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * self.ends[1]),
            np.pi / 2,
            line_types=[c, c],
            forbidden=True
        )
        lbc = StraightLane(
            lkb.position(self.ends[1], 0),
            lkb.position(self.ends[1], 0) + [self.ends[2], 0],
            line_types=[n, c],
            forbidden=True
        )

        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(self.ends[2], 0)))
        self.road = road

    def _make_hdv(self, num_HDV: int) -> None:
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        shift = float(self.config.get("hdv_spawn_shift_left", 0.0))

        spawn_points_world = (np.arange(270, 400, 9) - shift).tolist()
        available = spawn_points_world.copy()
        chosen = np.random.choice(available, num_HDV, replace=False).tolist()

        initial_speed = (np.random.rand(num_HDV) * 2 + 15).tolist()
        loc_noise = (np.random.rand(num_HDV) * 1 - 0.5).tolist()

        half = num_HDV // 2
        for _ in range(half):
            lane = road.network.get_lane(("a", "b", 0))
            start_x = float(getattr(lane, "start", [0.0, 0.0])[0])
            world_x = float(chosen.pop(0) + loc_noise.pop(0))
            world_x = max(world_x, 0.0)
            s_local = world_x - start_x
            x, y = lane.position(float(s_local), 0.0)
            v = other_vehicles_type(road, [float(x), float(y)], speed=float(initial_speed.pop(0)))
            v._is_controlled = False
            self._tag_hdv_once(v)
            road.vehicles.append(v)

        for _ in range(num_HDV - half):
            lane = road.network.get_lane(("b", "c", 0))
            start_x = float(getattr(lane, "start", [0.0, 0.0])[0])
            world_x = float(chosen.pop(0) + loc_noise.pop(0))
            s_local = world_x - start_x
            x, y = lane.position(float(s_local), 0.0)
            v = other_vehicles_type(road, [float(x), float(y)], speed=float(initial_speed.pop(0)))
            v._is_controlled = False
            self._tag_hdv_once(v)
            road.vehicles.append(v)

    def terminate(self):
        return

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds


register(id="merge-v41", entry_point=__name__ + ":MergeEnv")
