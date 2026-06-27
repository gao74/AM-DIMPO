import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agent.helpers import (cosine_beta_schedule, linear_beta_schedule, vp_beta_schedule, extract, Losses)
from agent.model import Model

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

class LatentDiffusion(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, noise_ratio,
                 beta_schedule='vp', n_timesteps=1000, loss_type='l2', clip_denoised=True, predict_epsilon=True):
        super(LatentDiffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.latent_dim = latent_dim
        self.noise_ratio = noise_ratio
        self.max_noise_ratio = noise_ratio
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        # 自动编码器
        self.autoencoder = AutoEncoder(action_dim, latent_dim)
        self.model = Model(state_dim, latent_dim)

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 缓存计算结果
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        self.loss_fn = Losses[loss_type]()

    def encode(self, action):
        return self.autoencoder.encode(action)

    # def decode(self, latent_action):
    #     return self.autoencoder.decode(latent_action)
    def decode(self, latent_action):
        if isinstance(latent_action, np.ndarray):
            latent_action = torch.tensor(latent_action, dtype=torch.float32)
        return self.autoencoder.decode(latent_action)

    def predict_start_from_noise(self, x_t, t, noise):
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_mean_variance(self, x, t, state):
        x_recon = self.predict_start_from_noise(x, t, self.model(x, t, state))
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        return x_recon, None, None  # Variance is not used directly

    @torch.no_grad()
    def p_sample(self, x, t, state):
        x_recon, _, _ = self.p_mean_variance(x, t, state)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))
        return x_recon + nonzero_mask * noise * self.noise_ratio

    @torch.no_grad()
    def p_sample_loop(self, state, shape):
        latent_action = torch.randn(shape, device=self.betas.device)
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((shape[0],), i, device=latent_action.device, dtype=torch.long)
            latent_action = self.p_sample(latent_action, timesteps, state)
        return latent_action

    @torch.no_grad()
    def sample(self, state, eval=False):
        self.noise_ratio = 0 if eval else self.max_noise_ratio
        latent_action = self.p_sample_loop(state, (state.shape[0], self.latent_dim))
        return self.decode(latent_action).clamp_(-1., 1.)

    def loss(self, actions, state):
        latent_actions = self.encode(actions)
        t = torch.randint(0, self.n_timesteps, (actions.shape[0],), device=actions.device).long()
        noise = torch.randn_like(latent_actions)
        latent_noisy = self.q_sample(latent_actions, t, noise)
        latent_recon = self.model(latent_noisy, t, state)
        return self.loss_fn(latent_recon, noise)

    def forward(self, state, eval=False):
        return self.sample(state, eval=eval)