import copy
import time
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from agent.model import Critic  # 注意：此处仅导入Critic，MLP类已本地覆盖
from agent.diffusion_ddim import Diffusion
from agent.vae import VAE
from agent.helpers import EMA
# ========== 新增自注意力模块 ==========
#带有注意力机制的扩散策略DIMPO算法
# class SelfAttention(nn.Module):
#     def __init__(self, feature_dim):
#         super(SelfAttention, self).__init__()
#         self.feature_dim = feature_dim
#         self.query = nn.Linear(feature_dim, feature_dim)
#         self.key = nn.Linear(feature_dim, feature_dim)
#         self.value = nn.Linear(feature_dim, feature_dim)
#         self.softmax = nn.Softmax(dim=-1)
# 
#     def forward(self, x):
#         q = self.query(x)
#         k = self.key(x)
#         v = self.value(x)
#         attention_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.feature_dim ** 0.5)
#         attention_weights = self.softmax(attention_scores)
#         out = torch.matmul(attention_weights, v)
#         return out + x  # 残差连接
# 
# 
# # ========== 覆盖原始MLP类 ==========
# class MLP(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(MLP, self).__init__()
#         self.l1 = nn.Linear(state_dim, 256)
#         self.l2 = nn.Linear(256, 256)
#         self.attention = SelfAttention(256)  # 插入自注意力
#         self.l3 = nn.Linear(256, action_dim)
# 
#     def forward(self, x):
#         x = F.relu(self.l1(x))
#         x = F.relu(self.l2(x))
#         x = self.attention(x)  # 应用自注意力
#         return torch.tanh(self.l3(x))  # 输出范围[-1,1]

class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim必须能被num_head整除"
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.out = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B = x.size(0)  # 批次维度
        # 分头处理
        q = self.query(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力权重
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn_weights = self.softmax(attn_scores)

        # 合并多头输出
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.head_dim)
        return self.out(out) + x  # 残差连接


# ========== 修改后的MLP类 ==========
class MLP(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 基础全连接层
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)

        # 层归一化与多头注意力
        self.norm = nn.LayerNorm(256)  # 新增层归一化
        self.attention = MultiHeadAttention(256)  # 替换为多头注意力

        # 输出层
        self.l3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.norm(x)  # 归一化后再输入注意力层
        x = self.attention(x)  # 应用多头注意力
        return torch.tanh(self.l3(x))


# ========== 原有DiPo类保持不变 ==========
class DiPo(object):
    def __init__(self, args, state_dim, action_space, memory, diffusion_memory, device):
        # ...（原有代码不变）
        action_dim = np.prod(action_space.shape)
        self.policy_type = args.policy_type
        # 当policy_type为MLP时，使用本地定义的带注意力MLP类
        if self.policy_type == 'Diffusion_DDIM':
            self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim,
                                   noise_ratio=args.noise_ratio,
                                   beta_schedule=args.beta_schedule,
                                   n_timesteps=args.n_timesteps).to(device)
        elif self.policy_type == 'VAE':
            self.actor = VAE(state_dim=state_dim, action_dim=action_dim,
                             device=device).to(device)
        else:
            self.actor = MLP(state_dim=state_dim, action_dim=action_dim).to(device)  # 使用新MLP

        # ...（后续代码不变）
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.diffusion_lr, eps=1e-5)

        self.memory = memory
        self.diffusion_memory = diffusion_memory
        self.action_gradient_steps = args.action_gradient_steps

        self.action_grad_norm = action_dim * args.ratio
        self.ac_grad_norm = args.ac_grad_norm

        self.step = 0
        self.tau = args.tau
        self.actor_target = copy.deepcopy(self.actor)
        self.update_actor_target_every = args.update_actor_target_every

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, eps=1e-5)

        self.action_dim = action_dim
        self.action_lr = args.action_lr
        self.device = device

        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = (action_space.high - action_space.low) / 2.
            self.action_bias = (action_space.high + action_space.low) / 2.

    def append_memory(self, state, action, reward, next_state, mask):
        action = (action - self.action_bias) / self.action_scale

        self.memory.append(state, action, reward, next_state, mask)
        self.diffusion_memory.append(state, action)

    def sample_action(self, state, eval=False, ddim=False, eta=0.0):
        """
        Sample an action using the actor (Diffusion/VAE/MLP).

        Parameters:
        - state: 当前状态
        - eval: 是否处于评估模式
        - ddpm1: 是否启用 DDIM 采样
        - eta: 控制 DDIM 随机性的参数（默认为 0，表示确定性采样）

        Returns:
        - 生成的动作
        """
        start_time = time.time()  # 记录开始时间
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        action = self.actor(state, eval=eval, ddim=ddim, eta=eta).cpu().data.numpy().flatten()
        action = action.clip(-1, 1)
        # 打印 self.action_scale 和 self.action_bias 的类型
        print(f"Type of self.action_scale: {type(self.action_scale)}")
        print(f"Type of self.action_bias: {type(self.action_bias)}")
        action = action * self.action_scale + self.action_bias
        # start_time = time.time()  # 记录开始时间
        latency = time.time() - start_time  # 计算延迟
        print(f"响应延迟: {latency:.4f}秒")
        return action

    def action_gradient(self, batch_size, log_writer):
        states, best_actions, idxs = self.diffusion_memory.sample(batch_size)

        actions_optim = torch.optim.Adam([best_actions], lr=self.action_lr, eps=1e-5)

        for i in range(self.action_gradient_steps):
            best_actions.requires_grad_(True)
            q1, q2 = self.critic(states, best_actions)
            loss = -torch.min(q1, q2)

            actions_optim.zero_grad()
            loss.backward(torch.ones_like(loss))

            if self.action_grad_norm > 0:
                nn.utils.clip_grad_norm_([best_actions], max_norm=self.action_grad_norm, norm_type=2)

            actions_optim.step()
            best_actions.requires_grad_(False)
            best_actions.clamp_(-1., 1.)

        best_actions = best_actions.detach()
        self.diffusion_memory.replace(idxs, best_actions.cpu().numpy())

        return states, best_actions

    def train(self, iterations, batch_size=256, log_writer=None):
        for _ in range(iterations):
            states, actions, rewards, next_states, masks = self.memory.sample(batch_size)

            """ Q Training """
            current_q1, current_q2 = self.critic(states, actions)

            next_actions = self.actor_target(next_states, eval=False)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            target_q = (rewards + masks * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()

            if self.ac_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.ac_grad_norm, norm_type=2)

            self.critic_optimizer.step()

            """ Policy Training """
            states, best_actions = self.action_gradient(batch_size, log_writer)

            actor_loss = self.actor.loss(best_actions, states)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            if self.ac_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.ac_grad_norm, norm_type=2)

            self.actor_optimizer.step()

            """ Step Target network """
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if self.step % self.update_actor_target_every == 0:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))

