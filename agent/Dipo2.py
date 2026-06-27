import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from agent.model import MLP, Critic
from agent.vae import VAE
from agent.diffusion_ldm import LatentDiffusion


class Latent(nn.Module):
    def __init__(self, action_dim, latent_dim):
        super(Latent, self).__init__()
        # 自动编码器
        self.encoder = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def encode(self, action):
        return self.encoder(action)

    def decode(self, latent_action):
        return self.decoder(latent_action)

    # def decode(self, latent_action):
    #     if isinstance(latent_action, np.ndarray):
    #         latent_action = torch.tensor(latent_action, dtype=torch.float32)
    #     return self.autoencoder.decode(latent_action)


class DiPo(object):
    def __init__(self, args, state_dim, action_space, memory, diffusion_memory, device):
        action_dim = np.prod(action_space.shape)
        latent_dim = args.latent_dim
        # latent_dim = 2  # 将 latent_dim 设置为 2

        self.device = device
        # self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.).to(self.device)
        # self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.).to(self.device)

        # 确保 action_scale 和 action_bias 是 PyTorch 张量，并在正确设备上
        self.action_lr = args.action_lr
        self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.).to(self.device)
        self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.).to(self.device)

        # print("Device of self.action_bias1:", self.action_bias.device)  # 打印 action_bias 的设备信息
        # print("Device of self.action_scale1:", self.action_scale.device)  # 打印 action_scale 的设备信息

        self.policy_type = args.policy_type
        if self.policy_type == 'LatentDiffusion':
            self.latent_diffusion = Latent(action_dim=action_dim, latent_dim=latent_dim).to(device)
            # self.actor = LatentDiffusion(state_dim=state_dim, action_dim=latent_dim,
            #                        noise_ratio=args.noise_ratio, beta_schedule=args.beta_schedule,
            #                        n_timesteps=args.n_timesteps).to(device)
            self.actor = LatentDiffusion(state_dim=state_dim, action_dim=action_dim, latent_dim=latent_dim,
                                         noise_ratio=args.noise_ratio, beta_schedule=args.beta_schedule,
                                         n_timesteps=args.n_timesteps).to(device)
        elif self.policy_type == 'VAE':
            self.actor = VAE(state_dim=state_dim, action_dim=action_dim, device=device).to(device)
        else:
            self.actor = MLP(state_dim=state_dim, action_dim=action_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.diffusion_lr, eps=1e-5)
        self.latent_diffusion_optimizer = torch.optim.Adam(self.latent_diffusion.parameters(), lr=args.vae_lr, eps=1e-5)

        self.memory = memory
        self.diffusion_memory = diffusion_memory
        self.action_gradient_steps = args.action_gradient_steps
        self.device = device

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, eps=1e-5)

        self.action_dim = action_dim
        # self.action_scale = (action_space.high - action_space.low) / 2.
        # self.action_bias = (action_space.high + action_space.low) / 2.
        # self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.).to(self.device)
        # self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.).to(self.device)

    # def append_memory(self, state, action, reward, next_state, mask):
    #     # Encode action before storing
    #     # latent_action = self.latent_diffusion.encode((action - self.action_bias) / self.action_scale)
    #     latent_action = self.latent_diffusion.encode(
    #         torch.FloatTensor((action - self.action_bias) / self.action_scale).to(self.device)
    #     )
    #     self.memory.append(state, latent_action, reward, next_state, mask)
    #     self.diffusion_memory.append(state, latent_action)
    # def append_memory(self, state, action, reward, next_state, mask):
    #     # 确保 action 是 PyTorch 张量
    #     action = torch.FloatTensor(action).to(self.device)
    #
    #     # 计算潜在动作
    #     latent_action = self.latent_diffusion.encode((action - self.action_bias) / self.action_scale)
    #
    #     # 存储到内存
    #     self.memory.append(state, latent_action, reward, next_state, mask)
    #     self.diffusion_memory.append(state, latent_action)

    # def append_memory(self, state, action, reward, next_state, mask):
    #     # 确保 action 是张量并位于 GPU 上
    #     action = torch.tensor(action, dtype=torch.float32).to(self.device)
    #     print("Device of action:", action.device)  # 打印 action 的设备信息
    #     print("Device of self.action_bias:", self.action_bias.device)  # 打印 latent_action 的设备信息
    #     print("Device of self.action_scale:",self.action_scale.device)  # 打印 latent_action 的设备信息
    #     # 计算潜在动作
    #     latent_action_gpu_1 = (action - self.action_bias) / self.action_scale
    #
    #     print("Device of latent_action_gpu_1:", latent_action_gpu_1.device)  # 打印 latent_action 的设备信息
    #     # latent_action = self.latent_diffusion.encode((action - self.action_bias) / self.action_scale)
    #     latent_action = self.latent_diffusion.encode(latent_action_gpu_1)
    #     print("Device of latent_action:", latent_action.device)  # 打印 latent_action 的设备信息
    #
    #     # 检查 latent_action 的数据
    #     print("latent_action values (GPU):", latent_action)  # 打印张量的值和设备信息
    #
    #     # 确保 latent_action 在 CPU 上
    #     latent_action = latent_action.cpu()
    #     print("Moved latent_action to CPU.")
    #     print("latent_action values (CPU):", latent_action)  # 再次打印以验证
    #
    #     # 如果需要 numpy 转换
    #     latent_action_numpy = latent_action.numpy()
    #     print("latent_action as numpy:", latent_action_numpy)
    #
    #     # 存储到内存
    #     self.memory.append(state, latent_action, reward, next_state, mask)
    #     self.diffusion_memory.append(state, latent_action)

    # def append_memory(self, state, action, reward, next_state, mask):
    #     # 确保 action 是张量并位于 GPU 上
    #     action = torch.tensor(action, dtype=torch.float32).to(self.device)
    #     print("Device of action:", action.device)  # 打印 action 的设备信息
    #
    #     # 计算潜在动作
    #     latent_action = self.latent_diffusion.encode((action - self.action_bias) / self.action_scale)
    #     print("Device of latent_action:", latent_action.device)  # 打印 latent_action 的设备信息
    #
    #     # 确保 latent_action 在 CPU 上
    #     latent_action = latent_action.cpu()
    #     print("Moved latent_action to CPU.")
    #
    #     # 如果需要 numpy 转换
    #     latent_action_numpy = latent_action.numpy()
    #     print("latent_action as numpy:", latent_action_numpy)
    #
    #     # 存储到内存
    #     self.memory.append(state, latent_action, reward, next_state, mask)
    #     self.diffusion_memory.append(state, latent_action)

    # def append_memory(self, state, action, reward, next_state, mask):
    #     # 确保 action 是张量并位于 GPU 上
    #     action = torch.tensor(action, dtype=torch.float32).to(self.device)
    #     print("Device of action:", action.device)  # 打印 action 的设备信息
    #     print("Device of self.action_bias:", self.action_bias.device)  # 打印 action_bias 的设备信息
    #     print("Device of self.action_scale:", self.action_scale.device)  # 打印 action_scale 的设备信息
    #
    #     # 计算潜在动作
    #     latent_action_gpu_1 = (action - self.action_bias) / self.action_scale
    #     print("Device of latent_action_gpu_1:", latent_action_gpu_1.device)  # 打印 latent_action 的设备信息
    #
    #     # 编码潜在动作
    #     latent_action = self.latent_diffusion.encode(latent_action_gpu_1)
    #     print("Device of latent_action:", latent_action.device)  # 打印 latent_action 的设备信息
    #
    #     # 检查 latent_action 的数据
    #     print("latent_action values (GPU):", latent_action)  # 打印张量的值和设备信息
    #
    #     # 确保 latent_action 在 CPU 上
    #     latent_action = latent_action.cpu()
    #     print("Moved latent_action to CPU.")
    #     print("latent_action values (CPU):", latent_action)  # 再次打印以验证
    #
    #     # 存储到内存
    #     self.memory.append(state, latent_action, reward, next_state, mask)
    #     self.diffusion_memory.append(state, latent_action)

    # def append_memory(self, state, action, reward, next_state, mask):
    #     # 确保 action 是张量并位于 GPU 上
    #     action = torch.tensor(action, dtype=torch.float32).to(self.device)
    #     print("Device of action:", action.device)
    #     print("Device of self.action_bias:", self.action_bias.device)
    #     print("Device of self.action_scale:", self.action_scale.device)
    #
    #     # 计算潜在动作
    #     latent_action_gpu_1 = (action - self.action_bias) / self.action_scale
    #     print("Device of latent_action_gpu_1:", latent_action_gpu_1.device)
    #
    #     # 编码潜在动作
    #     latent_action = self.latent_diffusion.encode(latent_action_gpu_1)
    #     print("Device of latent_action:", latent_action.device)
    #
    #     # 确保 latent_action 在 CPU 上，并移除梯度
    #     latent_action = latent_action.cpu().detach()
    #     print("Moved latent_action to CPU and detached gradients.")
    #     print("latent_action values (CPU, detached):", latent_action)
    #
    #     # 将 latent_action 转为 NumPy 数组
    #     latent_action_numpy = latent_action.numpy()
    #     print("latent_action as numpy:", latent_action_numpy)
    #
    #     # 存储到内存
    #     self.memory.append(state, latent_action_numpy, reward, next_state, mask)
    #     self.diffusion_memory.append(state, latent_action_numpy)

    import numpy as np

    def append_memory(self, state, action, reward, next_state, mask):
        # 确保 action 是张量并位于 GPU 上
        action = torch.tensor(action, dtype=torch.float32).to(self.device)
        # print("Device of action:", action.device)
        #
        # print("Device of self.action_bias:", self.action_bias.device)
        # print("Device of self.action_scale:", self.action_scale.device)

        # 计算潜在动作
        latent_action_gpu_1 = (action - self.action_bias) / self.action_scale
        # print("Device of latent_action_gpu_1:", latent_action_gpu_1.device)

        # 编码潜在动作
        latent_action = self.latent_diffusion.encode(latent_action_gpu_1)
        # print("Device of latent_action:", latent_action.device)

        # 确保 latent_action 在 CPU 上，并移除梯度
        latent_action = latent_action.cpu().detach()
        # print("Moved latent_action to CPU and detached gradients.")
        # print("latent_action values (CPU, detached):", latent_action)

        # 将 latent_action 转为 NumPy 数组
        latent_action_numpy = latent_action.numpy()
        # print("latent_action as numpy:", latent_action_numpy)

        # 检查 latent_action 的形状
        # print("Shape of latent_action_numpy:", latent_action_numpy.shape)

        # 使用池化操作将其降维为 (2,)
        if latent_action_numpy.shape != (2,):
            # 假设使用平均池化
            # 使用均值对 32 维数据进行降维到 2 维
            latent_action_numpy = np.mean(latent_action_numpy.reshape(-1, 16), axis=1)[:2]
            print("Adjusted latent_action_numpy shape using pooling:", latent_action_numpy.shape)

        # 存储到内存
        self.memory.append(state, latent_action_numpy, reward, next_state, mask)
        self.diffusion_memory.append(state, latent_action_numpy)

    # def sample_action(self, state, eval=False):
    #     state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
    #     latent_action = self.actor(state, eval).cpu().data.numpy().flatten()
    #     latent_action = torch.FloatTensor(latent_action).to(self.device)  # 转换 latent_action 为 Tensor
    #     print(f"Type of latent_action: {type(latent_action)}, Device: {latent_action.device}")
    #     action = self.latent_diffusion.decode(latent_action)
    #     # print("Device of self.action_bias2:", self.action_bias.device)  # 打印 action_bias 的设备信息
    #     action = action.clip(-1, 1) * self.action_scale + self.action_bias
    #     return action

    def sample_action(self, state, eval=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)  # 转换 state 为 Tensor
        latent_action = self.actor(state, eval).cpu().data.numpy().flatten()  # 转换为 NumPy 数组
        latent_action = torch.FloatTensor(latent_action).to(self.device)  # 转回 Tensor
        action = self.latent_diffusion.decode(latent_action)  # 调用 decode 方法
        if isinstance(action, torch.Tensor):  # 检查 action 是否为 Tensor
            action = action.cpu().detach().numpy()  # 先 detach，再转为 NumPy 数组
        action = np.clip(action, -1, 1)  # 进行 clip 操作，确保 action 是 numpy 数组
        action = action * self.action_scale.cpu().numpy() + self.action_bias.cpu().numpy()  # 确保 scale 和 bias 移动到 CPU 再转为 NumPy
        return action

    def action_gradient(self, batch_size, log_writer):
        states, latent_actions, idxs = self.diffusion_memory.sample(batch_size)
        actions_optim = torch.optim.Adam([latent_actions], lr=self.action_lr, eps=1e-5)

        for i in range(self.action_gradient_steps):
            latent_actions.requires_grad_(True)
            q1, q2 = self.critic(states, self.latent_diffusion.decode(latent_actions))
            loss = -torch.min(q1, q2)
            actions_optim.zero_grad()
            loss.backward(torch.ones_like(loss))
            actions_optim.step()
            latent_actions.requires_grad_(False)
            latent_actions.clamp_(-1., 1.)

        self.diffusion_memory.replace(idxs, latent_actions.cpu().numpy())
        return states, latent_actions

    def train(self, iterations, batch_size=256, log_writer=None):
        for _ in range(iterations):
            states, latent_actions, rewards, next_states, masks = self.memory.sample(batch_size)

            current_q1, current_q2 = self.critic(states, self.latent_diffusion.decode(latent_actions))
            # next_latent_actions = self.actor(next_states, eval=False)
            # next_latent_actions = self. actor_target(next_states, eval=False)
            next_latent_actions = self.actor_target(next_states)
            target_q1, target_q2 = self.critic_target(next_states, self.latent_diffusion.decode(next_latent_actions))
            target_q = torch.min(target_q1, target_q2)
            target_q = (rewards + masks * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            states, latent_actions = self.action_gradient(batch_size, log_writer)
            actor_loss = self.actor.loss(latent_actions, states)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

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
