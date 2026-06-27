
import argparse
import random
import sys
import time
import numpy as np
import torch
from gym.wrappers import RecordVideo
from agent.Dipo1 import DiPo
from agent.replay_memory import ReplayMemory, DiffusionMemory
from tensorboardX import SummaryWriter
import gym
import os
import pickle
import datetime
from collections import deque
import matplotlib.pyplot as plt

sys.path.append("../highway-env")
import highway_env
"""
AM-DIMPO: Action-Mask-Guided Safe Diffusion-Implicit Policy Optimization
===========================================================================

This file is a complete, heavily commented reference implementation written
from the uploaded paper and adapted to the uploaded MergeEnv experimental
environment.

Important notes
---------------
1. The paper clearly specifies the core algorithmic pipeline, the structured
   state design, the DDIM-based action generation, the action mask, the
   safety-internalization loss, and the main hyperparameters.
2. Some implementation-level details are *not fully specified* in the paper
   (for example, exact critic-guidance coefficient, exact mapping from action
   range to physical steering angle, and some simulator-specific geometry
   constants). Those parts are implemented here in a reasonable, explicit,
   and easily editable way.
3. This script is designed to work with the uploaded environment file that
   registers `merge-v33`. Put that environment file in the Python path or the
   same working directory before running.

What this file contains
-----------------------
- Structured state extractor matching the paper.
- Diffusion actor with DDIM sampling.
- Twin critics + target networks.
- State-dependent continuous action mask.
- Safety-internalization loss.
- Replay buffer.
- Full training loop.
- Evaluation loop with mask-on / mask-off options.
- Density-aware DDIM step switching.

Recommended software stack from the paper
-----------------------------------------
Python 3.10, PyTorch 2.1.0, Gym 0.26.2, NumPy 1.24.4.
"""

from __future__ import annotations

import copy
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

# -----------------------------------------------------------------------------
# 0. User environment import
# -----------------------------------------------------------------------------
# The uploaded experimental environment registers `merge-v33` at import time.
# Adjust the import below if your local file name is different.
try:
    import ef4155e0_def5_4e69_833b_130d7512f7ac as merge_env_module  # type: ignore
except Exception:
    # Fallback: if the environment file has been renamed, importing it is not
    # strictly required here as long as `merge-v33` was already registered.
    merge_env_module = None


# -----------------------------------------------------------------------------
# 1. Reproducibility helpers
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Set all common random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# 2. Configuration
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # Environment
    env_id: str = "merge-v33"
    train_steps: int = 2_000_000
    max_episode_steps: int = 100
    eval_every_episodes: int = 200
    eval_episodes: int = 20
    robustness_eval_episodes: int = 500
    traffic_density_train: int = 4     # Uploaded env uses discrete density level.
    easy_density: int = 2
    hard_density: int = 4

    # RL / optimization
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 256
    replay_size: int = 1_000_000
    tau: float = 0.005
    warmup_steps: int = 10_000
    updates_per_step: int = 1
    gradient_clip_norm: float = 1.0

    # Network
    hidden_dims: Tuple[int, int, int] = (256, 256, 256)

    # Diffusion / DDIM
    diffusion_train_steps: int = 40      # T in the paper's default configuration.
    ddim_sample_steps_easy: int = 20     # Density-aware inference from Table 5.
    ddim_sample_steps_hard: int = 40
    ddim_eta: float = 0.03
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # Safety / AM-DIMPO
    lambda_safe: float = 0.05
    action_guidance_step: float = 0.05   # Critic-gradient refinement; inferred.
    action_low: float = -1.0
    action_high: float = 1.0
    steering_soft_limit: float = 0.25    # Paper explicitly gives example steering bound.
    density_switch_threshold: int = 5    # Vehicles within 100 m.

    # State extraction
    nearby_k: int = 6
    density_back_range: float = 40.0
    density_front_range: float = 60.0
    gap_search_range: float = 120.0

    # Device / logging
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 10
    save_dir: str = "./outputs_am_dimpo"
    checkpoint_name: str = "am_dimpo_best.pt"


# -----------------------------------------------------------------------------
# 3. Replay buffer
# -----------------------------------------------------------------------------
class ReplayBuffer:
    """A simple NumPy replay buffer for off-policy training."""

    def __init__(self, state_dim: int, action_dim: int, capacity: int) -> None:
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "state": torch.tensor(self.states[idx], device=device),
            "action": torch.tensor(self.actions[idx], device=device),
            "reward": torch.tensor(self.rewards[idx], device=device),
            "next_state": torch.tensor(self.next_states[idx], device=device),
            "done": torch.tensor(self.dones[idx], device=device),
        }
        return batch

    def __len__(self) -> int:
        return self.size


# -----------------------------------------------------------------------------
# 4. Neural network building blocks
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    """Standard ReLU MLP used by both actor and critic, matching the paper."""

    def __init__(self, in_dim: int, hidden_dims: Sequence[int], out_dim: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinusoidalTimeEmbedding(nn.Module):
    """Classic sinusoidal embedding for diffusion timestep conditioning."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        freq = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, device=device) / max(half_dim - 1, 1)
        )
        angles = t.float().unsqueeze(-1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# -----------------------------------------------------------------------------
# 5. Diffusion actor
# -----------------------------------------------------------------------------
class DiffusionActor(nn.Module):
    """
    Diffusion actor epsilon_phi(a_k, s, k).

    Input:
        - state s
        - noisy action a_k
        - timestep k
    Output:
        - predicted noise epsilon
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        diffusion_steps: int,
        action_low: float,
        action_high: float,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.diffusion_steps = diffusion_steps
        self.action_low = action_low
        self.action_high = action_high

        self.time_embed = SinusoidalTimeEmbedding(64)
        self.net = MLP(state_dim + action_dim + 64, hidden_dims, action_dim)

    def forward(self, state: torch.Tensor, noisy_action: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(timestep)
        x = torch.cat([state, noisy_action, t_emb], dim=-1)
        return self.net(x)


# -----------------------------------------------------------------------------
# 6. Twin critics
# -----------------------------------------------------------------------------
class TwinCritic(nn.Module):
    """Two Q networks for more stable off-policy learning."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        in_dim = state_dim + action_dim
        self.q1 = MLP(in_dim, hidden_dims, 1)
        self.q2 = MLP(in_dim, hidden_dims, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)


# -----------------------------------------------------------------------------
# 7. Diffusion utilities
# -----------------------------------------------------------------------------
class DiffusionSchedule:
    """Precomputes all scalar diffusion coefficients used in q and DDIM sampling."""

    def __init__(self, diffusion_steps: int, beta_start: float, beta_end: float, device: torch.device) -> None:
        self.diffusion_steps = diffusion_steps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, diffusion_steps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        # Timesteps are sampled uniformly from 1..T in the paper.
        return torch.randint(1, self.diffusion_steps + 1, (batch_size,), device=self.device)

    def gather(self, arr: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Paper notation is 1-based, tensor indexing is 0-based.
        return arr[t - 1].unsqueeze(-1)


# -----------------------------------------------------------------------------
# 8. State extraction and environment wrapper
# -----------------------------------------------------------------------------
class MergeStateExtractor:
    """
    Converts the raw simulator state into the structured paper-style state:
        [ego, geometry, density, gap, nearby vehicles]

    Paper structure:
        s_t = [s_ego, s_geo, s_dens, s_gap, {s_nbr_i}_{i=1..K}]

    This wrapper uses `env.unwrapped.road` and `env.unwrapped.controlled_vehicles[0]`
    from the uploaded highway-env based simulator.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.state_dim = 3 + 2 + 1 + 6 + cfg.nearby_k * 4

    def _get_ego(self, env: gym.Env):
        env0 = env.unwrapped
        return env0.controlled_vehicles[0]

    def _get_all_vehicles(self, env: gym.Env):
        env0 = env.unwrapped
        return [v for v in env0.road.vehicles if hasattr(v, "position")]

    def _lane_center_offset(self, env: gym.Env, vehicle) -> float:
        lane = env.unwrapped.road.network.get_lane(vehicle.lane_index)
        _, lat = lane.local_coordinates(vehicle.position)
        return float(lat)

    def _remaining_distance(self, vehicle) -> float:
        # In the uploaded env, merge success is roughly checked at x > 440.
        return max(0.0, 440.0 - float(vehicle.position[0]))

    def _merge_indicator(self, vehicle) -> float:
        # Consistent with the paper's merged indicator zeta(s_t).
        return float(vehicle.lane_index in [("b", "c", 0), ("c", "d", 0)])

    def _local_density(self, env: gym.Env, ego) -> Tuple[float, int]:
        x = float(ego.position[0])
        left = x - self.cfg.density_back_range
        right = x + self.cfg.density_front_range
        count = 0
        for v in self._get_all_vehicles(env):
            if v is ego:
                continue
            vx = float(v.position[0])
            if left <= vx <= right:
                count += 1
        density = count / (self.cfg.density_back_range + self.cfg.density_front_range)
        return float(density), count

    def _candidate_target_lanes(self) -> List[Tuple[str, str, int]]:
        return [("b", "c", 0), ("c", "d", 0)]

    def _gap_features(self, env: gym.Env, ego) -> np.ndarray:
        target_lanes = set(self._candidate_target_lanes())
        ego_x = float(ego.position[0])
        ego_v = float(ego.speed)
        ego_a = float(getattr(ego, "acceleration", 0.0))

        front = None
        rear = None
        front_dx = float("inf")
        rear_dx = float("inf")

        for v in self._get_all_vehicles(env):
            if v is ego:
                continue
            if getattr(v, "lane_index", None) not in target_lanes:
                continue
            dx = float(v.position[0]) - ego_x
            if 0.0 <= dx < front_dx:
                front_dx = dx
                front = v
            if dx < 0.0 and abs(dx) < rear_dx:
                rear_dx = abs(dx)
                rear = v

        # Large default gap means "no relevant vehicle found".
        df = min(front_dx, self.cfg.gap_search_range) if front is not None else self.cfg.gap_search_range
        dr = min(rear_dx, self.cfg.gap_search_range) if rear is not None else self.cfg.gap_search_range

        vf = float(front.speed) if front is not None else ego_v
        vr = float(rear.speed) if rear is not None else ego_v
        af = float(getattr(front, "acceleration", 0.0)) if front is not None else 0.0
        ar = float(getattr(rear, "acceleration", 0.0)) if rear is not None else 0.0

        dvf = ego_v - vf
        dvr = ego_v - vr
        daf = ego_a - af
        dar = ego_a - ar
        return np.array([df, dr, dvf, dvr, daf, dar], dtype=np.float32)

    def _nearby_vehicle_features(self, env: gym.Env, ego) -> np.ndarray:
        ego_x, ego_y = float(ego.position[0]), float(ego.position[1])
        ego_vx = float(ego.speed * math.cos(float(getattr(ego, "heading", 0.0))))
        ego_vy = float(ego.speed * math.sin(float(getattr(ego, "heading", 0.0))))

        features: List[List[float]] = []
        for v in self._get_all_vehicles(env):
            if v is ego:
                continue
            dx = float(v.position[0]) - ego_x
            dy = float(v.position[1]) - ego_y
            dist = math.hypot(dx, dy)
            vx = float(v.speed * math.cos(float(getattr(v, "heading", 0.0))))
            vy = float(v.speed * math.sin(float(getattr(v, "heading", 0.0))))
            features.append([dist, dx, dy, vx - ego_vx, vy - ego_vy])

        # Sort by Euclidean distance to keep the nearest K vehicles.
        features.sort(key=lambda item: item[0])

        nbr = []
        for item in features[: self.cfg.nearby_k]:
            _, dx, dy, dvx, dvy = item
            nbr.append([dx, dy, dvx, dvy])

        while len(nbr) < self.cfg.nearby_k:
            nbr.append([0.0, 0.0, 0.0, 0.0])

        return np.asarray(nbr, dtype=np.float32).reshape(-1)

    def extract(self, env: gym.Env) -> np.ndarray:
        ego = self._get_ego(env)
        ey = self._lane_center_offset(env, ego)
        ego_state = np.array(
            [
                float(ego.speed),
                float(getattr(ego, "acceleration", 0.0)),
                ey,
            ],
            dtype=np.float32,
        )

        geo_state = np.array(
            [
                self._remaining_distance(ego),
                self._merge_indicator(ego),
            ],
            dtype=np.float32,
        )

        rho_loc, surrounding_count = self._local_density(env, ego)
        dens_state = np.array([rho_loc], dtype=np.float32)

        gap_state = self._gap_features(env, ego)
        nbr_state = self._nearby_vehicle_features(env, ego)

        state = np.concatenate([ego_state, geo_state, dens_state, gap_state, nbr_state], axis=0)
        assert state.shape[0] == self.state_dim
        return state.astype(np.float32)

    def surrounding_vehicle_count_100m(self, env: gym.Env) -> int:
        ego = self._get_ego(env)
        ego_x = float(ego.position[0])
        count = 0
        for v in self._get_all_vehicles(env):
            if v is ego:
                continue
            if abs(float(v.position[0]) - ego_x) <= 100.0:
                count += 1
        return count


class StructuredStateWrapper(gym.Wrapper):
    """Gym wrapper that replaces the default observation with the paper-style state vector."""

    def __init__(self, env: gym.Env, extractor: MergeStateExtractor):
        super().__init__(env)
        self.extractor = extractor
        low = np.full((extractor.state_dim,), -np.inf, dtype=np.float32)
        high = np.full((extractor.state_dim,), np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            _, info = result
        else:
            info = {}
        obs = self.extractor.extract(self.env)
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            _, reward, terminated, truncated, info = result
        else:
            _, reward, done, info = result
            terminated, truncated = done, False
        obs = self.extractor.extract(self.env)
        return obs, reward, terminated, truncated, info


# -----------------------------------------------------------------------------
# 9. Action mask
# -----------------------------------------------------------------------------
class ActionMask:
    """
    Continuous state-dependent action mask F(s, a_raw) -> a_safe.

    The paper states that the feasible set is approximated by state-dependent
    lower and upper bounds and implemented by clipping. It also gives the
    example that steering may be limited to +/- 0.25.

    Below, the longitudinal bound is adapted from speed and road-end logic,
    while the steering bound is narrowed when the vehicle is already close to
    the main lane center. These are simulator-aware heuristics that keep the
    implementation explicit and editable.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg

    def bounds_from_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Derive a_min(s), a_max(s) from the paper-style state.

        State layout:
            [v_e, a_e, e_y, d_rem, chi_merge, rho_loc, df, dr, ...]
        """
        v_e = state[:, 0]
        e_y = state[:, 2]
        d_rem = state[:, 3]
        chi_merge = state[:, 4]

        batch = state.shape[0]
        a_min = torch.full((batch, 2), self.cfg.action_low, device=state.device)
        a_max = torch.full((batch, 2), self.cfg.action_high, device=state.device)

        # ---------------- Longitudinal channel bounds ----------------
        # Heuristic 1: if speed is already high, suppress further acceleration.
        a_max[:, 0] = torch.where(v_e > 19.0, torch.full_like(v_e, 0.2), a_max[:, 0])

        # Heuristic 2: if speed is very low, do not allow large deceleration.
        a_min[:, 0] = torch.where(v_e < 2.0, torch.full_like(v_e, -0.1), a_min[:, 0])

        # Heuristic 3: close to ramp end and still not merged -> discourage braking.
        close_to_end = (d_rem < 40.0) & (chi_merge < 0.5)
        a_min[:, 0] = torch.where(close_to_end, torch.maximum(a_min[:, 0], torch.tensor(-0.2, device=state.device)), a_min[:, 0])

        # ---------------- Lateral channel bounds ----------------
        steer_limit = torch.full_like(v_e, self.cfg.steering_soft_limit)

        # If vehicle is almost centered after merging, reduce steering range to help stability.
        settled_mainline = (chi_merge > 0.5) & (torch.abs(e_y) < 0.3)
        steer_limit = torch.where(settled_mainline, torch.full_like(steer_limit, 0.12), steer_limit)

        # Close to end of ramp and not merged -> allow slightly larger steering authority.
        urgent_merge = (chi_merge < 0.5) & (d_rem < 30.0)
        steer_limit = torch.where(urgent_merge, torch.full_like(steer_limit, 0.35), steer_limit)

        a_min[:, 1] = -steer_limit
        a_max[:, 1] = steer_limit
        return a_min, a_max

    def apply(self, raw_action: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a_min, a_max = self.bounds_from_state(state)
        safe_action = torch.max(torch.min(raw_action, a_max), a_min)
        correction = safe_action - raw_action
        return safe_action, a_min, a_max


# -----------------------------------------------------------------------------
# 10. AM-DIMPO agent
# -----------------------------------------------------------------------------
class AMDIMPOAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Core modules
        self.actor = DiffusionActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=cfg.hidden_dims,
            diffusion_steps=cfg.diffusion_train_steps,
            action_low=cfg.action_low,
            action_high=cfg.action_high,
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)

        self.critic = TwinCritic(state_dim, action_dim, cfg.hidden_dims).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        self.schedule = DiffusionSchedule(
            diffusion_steps=cfg.diffusion_train_steps,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            device=self.device,
        )
        self.mask = ActionMask(cfg)

    # ---------------- Diffusion training utilities ----------------
    def q_sample(self, clean_action: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.schedule.gather(self.schedule.sqrt_alpha_bars, t)
        sqrt_1mab = self.schedule.gather(self.schedule.sqrt_one_minus_alpha_bars, t)
        return sqrt_ab * clean_action + sqrt_1mab * noise

    def predict_x0(self, state: torch.Tensor, noisy_action: torch.Tensor, t: torch.Tensor, actor: Optional[nn.Module] = None) -> torch.Tensor:
        model = self.actor if actor is None else actor
        eps_pred = model(state, noisy_action, t)
        sqrt_ab = self.schedule.gather(self.schedule.sqrt_alpha_bars, t)
        sqrt_1mab = self.schedule.gather(self.schedule.sqrt_one_minus_alpha_bars, t)
        x0 = (noisy_action - sqrt_1mab * eps_pred) / (sqrt_ab + 1e-8)
        return x0

    def diffusion_loss(self, state: torch.Tensor, clean_action: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        t = self.schedule.sample_timesteps(batch_size)
        noise = torch.randn_like(clean_action)
        noisy_action = self.q_sample(clean_action, t, noise)
        noise_pred = self.actor(state, noisy_action, t)
        return F.mse_loss(noise_pred, noise)

    # ---------------- DDIM sampling ----------------
    def get_ddim_timesteps(self, sample_steps: int) -> List[int]:
        # Uniformly spaced subsampling from the full diffusion chain, as in Table 5.
        T = self.cfg.diffusion_train_steps
        if sample_steps >= T:
            return list(range(T, 0, -1))
        indices = np.linspace(T, 1, sample_steps, dtype=int).tolist()
        # Remove duplicates that can appear due to integer rounding.
        unique = []
        for idx in indices:
            if idx not in unique:
                unique.append(idx)
        return unique

    @torch.no_grad()
    def sample_raw_action(
        self,
        state: torch.Tensor,
        sample_steps: int,
        eta: float,
        use_target: bool = False,
    ) -> torch.Tensor:
        """
        DDIM-style implicit sampler for the action policy.

        This follows the paper's idea directly:
        - Start from Gaussian noise a_K ~ N(0, I).
        - Run reverse denoising over a reduced timestep set.
        - Use small eta > 0 to preserve limited stochasticity.
        """
        actor = self.actor_target if use_target else self.actor
        batch_size = state.shape[0]
        x = torch.randn(batch_size, self.action_dim, device=self.device)

        timesteps = self.get_ddim_timesteps(sample_steps)
        for i, t_val in enumerate(timesteps):
            t = torch.full((batch_size,), t_val, device=self.device, dtype=torch.long)
            x0_pred = self.predict_x0(state, x, t, actor=actor)
            eps_pred = actor(state, x, t)

            alpha_bar_t = self.schedule.alpha_bars[t_val - 1]
            if i == len(timesteps) - 1:
                alpha_bar_prev = torch.tensor(1.0, device=self.device)
            else:
                t_prev = timesteps[i + 1]
                alpha_bar_prev = self.schedule.alpha_bars[t_prev - 1]

            sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

            # Standard DDIM update written in a numerically stable form.
            dir_term = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma_t ** 2, min=0.0)) * eps_pred
            x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_term + sigma_t * noise

        return torch.clamp(x, self.cfg.action_low, self.cfg.action_high)

    def refine_with_critic(self, state: torch.Tensor, raw_action: torch.Tensor) -> torch.Tensor:
        """
        Implements the paper's action-gradient refinement:
            a <- a + eta_a * grad_a Q(s, a)

        We use Q1 only for refinement to keep the gradient path simple.
        """
        action_var = raw_action.clone().detach().requires_grad_(True)
        q = self.critic.q1_only(state, action_var).sum()
        grad = torch.autograd.grad(q, action_var, retain_graph=False, create_graph=False)[0]
        refined = action_var + self.cfg.action_guidance_step * grad
        return torch.clamp(refined.detach(), self.cfg.action_low, self.cfg.action_high)

    @torch.no_grad()
    def act(self, state_np: np.ndarray, sample_steps: int, use_mask: bool = True) -> Tuple[np.ndarray, Dict[str, float]]:
        state = torch.tensor(state_np, device=self.device).unsqueeze(0)
        raw = self.sample_raw_action(state, sample_steps=sample_steps, eta=self.cfg.ddim_eta, use_target=False)
        # Critic guidance requires gradients, so we temporarily enable grad.
        with torch.enable_grad():
            raw_refined = self.refine_with_critic(state, raw)

        info: Dict[str, float] = {}
        if use_mask:
            safe, _, _ = self.mask.apply(raw_refined, state)
            correction = torch.norm(safe - raw_refined, dim=-1).mean().item()
            violation = ((raw_refined < -1.0) | (raw_refined > 1.0)).float().mean().item()
            info["correction_norm"] = correction
            info["raw_violation_rate"] = violation
            return safe.squeeze(0).cpu().numpy().astype(np.float32), info
        return raw_refined.squeeze(0).cpu().numpy().astype(np.float32), info

    # ---------------- Training step ----------------
    def update(self, batch: Dict[str, torch.Tensor], sample_steps_for_target: int) -> Dict[str, float]:
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        next_state = batch["next_state"]
        done = batch["done"]

        # ----- Critic update -----
        with torch.no_grad():
            next_raw = self.sample_raw_action(
                next_state,
                sample_steps=sample_steps_for_target,
                eta=self.cfg.ddim_eta,
                use_target=True,
            )
            next_safe, _, _ = self.mask.apply(next_raw, next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_safe)
            target_q = torch.min(target_q1, target_q2)
            y = reward + self.cfg.gamma * (1.0 - done) * target_q

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), self.cfg.gradient_clip_norm)
        self.critic_optim.step()

        # ----- Actor update -----
        # 1) Diffusion denoising loss on replay actions.
        l_diff = self.diffusion_loss(state, action)

        # 2) Safety-internalization loss.
        sampled_raw = self.sample_raw_action(
            state,
            sample_steps=sample_steps_for_target,
            eta=self.cfg.ddim_eta,
            use_target=False,
        )
        sampled_safe, _, _ = self.mask.apply(sampled_raw, state)
        l_safe = ((sampled_raw - sampled_safe) ** 2).sum(dim=-1).mean()

        actor_loss = l_diff + self.cfg.lambda_safe * l_safe

        self.actor_optim.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), self.cfg.gradient_clip_norm)
        self.actor_optim.step()

        # ----- Soft target update -----
        self.soft_update(self.critic, self.critic_target, self.cfg.tau)
        self.soft_update(self.actor, self.actor_target, self.cfg.tau)

        # ----- Monitoring metrics -----
        with torch.no_grad():
            mask_activation_rate = (torch.abs(sampled_raw - sampled_safe).sum(dim=-1) > 1e-6).float().mean().item()
            correction_magnitude = torch.norm(sampled_raw - sampled_safe, dim=-1).mean().item()
            a_min, a_max = self.mask.bounds_from_state(state)
            violation_rate = ((sampled_raw < a_min) | (sampled_raw > a_max)).any(dim=-1).float().mean().item()

        return {
            "critic_loss": float(critic_loss.item()),
            "diffusion_loss": float(l_diff.item()),
            "safe_loss": float(l_safe.item()),
            "actor_loss": float(actor_loss.item()),
            "mask_activation_rate": float(mask_activation_rate),
            "correction_magnitude": float(correction_magnitude),
            "raw_violation_rate": float(violation_rate),
        }

    @staticmethod
    def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1.0 - tau) * tgt_param.data)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
                "cfg": asdict(self.cfg),
            },
            path,
        )


# -----------------------------------------------------------------------------
# 11. Training / evaluation helpers
# -----------------------------------------------------------------------------
def build_env(cfg: TrainConfig, traffic_density: int, seed: Optional[int] = None) -> Tuple[gym.Env, MergeStateExtractor]:
    env = gym.make(cfg.env_id)

    # Apply the density level used by the uploaded environment file.
    if hasattr(env.unwrapped, "config"):
        env.unwrapped.config["traffic_density"] = traffic_density
        env.unwrapped.config["duration"] = cfg.max_episode_steps / max(getattr(env.unwrapped, "config", {}).get("policy_frequency", 5), 1)

    if seed is not None:
        try:
            env.reset(seed=seed)
            env.action_space.seed(seed)
        except Exception:
            pass

    extractor = MergeStateExtractor(cfg)
    env = StructuredStateWrapper(env, extractor)
    return env, extractor


def select_density_aware_steps(cfg: TrainConfig, extractor: MergeStateExtractor, env: gym.Env) -> int:
    count_100m = extractor.surrounding_vehicle_count_100m(env)
    if count_100m >= cfg.density_switch_threshold:
        return cfg.ddim_sample_steps_hard
    return cfg.ddim_sample_steps_easy


def evaluate_policy(
    agent: AMDIMPOAgent,
    cfg: TrainConfig,
    traffic_density: int,
    episodes: int,
    seed_offset: int = 0,
    use_mask: bool = True,
) -> Dict[str, float]:
    env, extractor = build_env(cfg, traffic_density=traffic_density, seed=cfg.seed + seed_offset)

    returns = []
    success = 0
    collision = 0
    merge_times = []
    avg_speeds = []
    mask_activation_flags = []
    correction_norms = []

    for ep in range(episodes):
        state, _ = env.reset(seed=cfg.seed + seed_offset + ep)
        done = False
        total_reward = 0.0
        steps = 0
        speed_sum = 0.0

        while not done:
            ddim_steps = select_density_aware_steps(cfg, extractor, env)
            action, info = agent.act(state, sample_steps=ddim_steps, use_mask=use_mask)

            if use_mask:
                correction_norms.append(float(info.get("correction_norm", 0.0)))
                mask_activation_flags.append(1.0 if info.get("correction_norm", 0.0) > 1e-6 else 0.0)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            total_reward += float(reward)
            steps += 1

            ego = env.unwrapped.controlled_vehicles[0]
            speed_sum += float(ego.speed)
            state = next_state

            if done:
                crashed = bool(getattr(ego, "crashed", False))
                reached = bool(float(ego.position[0]) > 440.0 and ego.lane_index in [("b", "c", 0), ("c", "d", 0)])
                if crashed:
                    collision += 1
                if reached:
                    success += 1
                merge_times.append(float(steps / max(getattr(env.unwrapped, "config", {}).get("policy_frequency", 5), 1)))
                avg_speeds.append(speed_sum / max(steps, 1))
                returns.append(total_reward)

    env.close()
    metrics = {
        "success_rate": success / episodes,
        "collision_rate": collision / episodes,
        "avg_reward": float(np.mean(returns)) if returns else 0.0,
        "avg_merge_time": float(np.mean(merge_times)) if merge_times else 0.0,
        "avg_speed": float(np.mean(avg_speeds)) if avg_speeds else 0.0,
    }
    if use_mask:
        metrics["mask_activation_rate"] = float(np.mean(mask_activation_flags)) if mask_activation_flags else 0.0
        metrics["mean_correction_magnitude"] = float(np.mean(correction_norms)) if correction_norms else 0.0
    return metrics


# -----------------------------------------------------------------------------
# 12. Main training loop
# -----------------------------------------------------------------------------
def train_am_dimpo(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    env, extractor = build_env(cfg, traffic_density=cfg.traffic_density_train, seed=cfg.seed)
    state_dim = extractor.state_dim
    action_dim = int(np.prod(env.action_space.shape))

    agent = AMDIMPOAgent(state_dim=state_dim, action_dim=action_dim, cfg=cfg)
    buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, capacity=cfg.replay_size)

    state, _ = env.reset(seed=cfg.seed)
    episode = 0
    episode_reward = 0.0
    episode_steps = 0
    best_success_rate = -1.0
    training_start_time = time.time()

    print("=" * 88)
    print("Start training AM-DIMPO")
    print(f"state_dim={state_dim}, action_dim={action_dim}, device={cfg.device}")
    print("=" * 88)

    for global_step in range(1, cfg.train_steps + 1):
        episode_steps += 1

        # Density-aware DDIM step switching from Table 5.
        ddim_steps = select_density_aware_steps(cfg, extractor, env)

        if global_step < cfg.warmup_steps:
            action = env.action_space.sample().astype(np.float32)
        else:
            action, _ = agent.act(state, sample_steps=ddim_steps, use_mask=True)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        buffer.add(state, action, reward, next_state, done)

        state = next_state
        episode_reward += float(reward)

        # Update after enough data is collected.
        if len(buffer) >= cfg.batch_size and global_step >= cfg.warmup_steps:
            for _ in range(cfg.updates_per_step):
                batch = buffer.sample(cfg.batch_size, device=agent.device)
                train_info = agent.update(batch, sample_steps_for_target=ddim_steps)
        else:
            train_info = {}

        if done or episode_steps >= cfg.max_episode_steps:
            episode += 1
            if episode % cfg.log_interval == 0:
                elapsed = time.time() - training_start_time
                print(
                    f"Episode {episode:6d} | Step {global_step:8d} | "
                    f"EpReward {episode_reward:8.3f} | "
                    f"Buffer {len(buffer):8d} | Elapsed {elapsed/60.0:7.2f} min"
                )
                if train_info:
                    print(
                        "  TrainInfo: "
                        + ", ".join([f"{k}={v:.5f}" for k, v in train_info.items()])
                    )

            state, _ = env.reset()
            episode_reward = 0.0
            episode_steps = 0

            # Periodic evaluation, matching the paper's protocol idea.
            if episode > 0 and episode % cfg.eval_every_episodes == 0 and len(buffer) >= cfg.batch_size:
                eval_metrics = evaluate_policy(
                    agent,
                    cfg,
                    traffic_density=cfg.hard_density,
                    episodes=cfg.eval_episodes,
                    seed_offset=10_000 + episode,
                    use_mask=True,
                )
                print("-" * 88)
                print(f"Evaluation after episode {episode}")
                print(
                    ", ".join(
                        [f"{k}={v:.4f}" for k, v in eval_metrics.items()]
                    )
                )
                print("-" * 88)

                if eval_metrics["success_rate"] > best_success_rate:
                    best_success_rate = eval_metrics["success_rate"]
                    save_path = os.path.join(cfg.save_dir, cfg.checkpoint_name)
                    agent.save(save_path)
                    print(f"Saved new best model to: {save_path}")

    env.close()

    # Final mask-on and mask-off evaluation.
    print("=" * 88)
    print("Final evaluation")
    final_on = evaluate_policy(agent, cfg, traffic_density=cfg.hard_density, episodes=cfg.eval_episodes, seed_offset=20_000, use_mask=True)
    final_off = evaluate_policy(agent, cfg, traffic_density=cfg.hard_density, episodes=cfg.eval_episodes, seed_offset=30_000, use_mask=False)
    print("Mask-ON :", final_on)
    print("Mask-OFF:", final_off)
    print("=" * 88)


# -----------------------------------------------------------------------------
# 13. Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = TrainConfig()
    print("Training config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    train_am_dimpo(cfg)
