import torch  # 导入 PyTorch 库，用于张量计算
from torch import nn  # 从 PyTorch 导入神经网络模块
from torch.nn import functional as F  # 导入常用的函数模块，用于计算损失等操作
from agent.helpers import init_weights  # 从辅助模块中导入自定义的初始化函数，用于初始化权重


class VAE(nn.Module):  # 定义变分自编码器类，继承 nn.Module 基类
    def __init__(self, state_dim, action_dim, device, hidden_size=256) -> None:  # 初始化函数，定义 VAE 的参数
        super(VAE, self).__init__()  # 调用父类的初始化方法

        self.hidden_size = hidden_size  # 设置隐藏层的维度大小，默认为 256
        self.action_dim = action_dim  # 设置动作空间的维度大小

        input_dim = state_dim + action_dim  # 输入维度为状态和动作维度之和

        # 编码器，用于将输入映射到潜在空间
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),  # 第一层线性层，将输入映射到隐藏层大小
            nn.Mish(),  # Mish 激活函数，增加非线性
            nn.Linear(hidden_size, hidden_size),  # 第二层线性层，维度保持不变
            nn.Mish(),  # 再次使用 Mish 激活函数
            nn.Linear(hidden_size, hidden_size),  # 第三层线性层，将输出维度保持为隐藏层大小
            nn.Mish()  # 使用 Mish 激活函数，使输出具有更强的非线性
        )

        # 用于生成潜在变量 z 的均值向量
        self.fc_mu = nn.Linear(hidden_size, hidden_size)  # 定义线性层，输出均值

        # 用于生成潜在变量 z 的对数方差向量
        self.fc_var = nn.Linear(hidden_size, hidden_size)  # 定义线性层，输出对数方差

        # 解码器，用于将潜在变量和状态映射回动作空间
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size + state_dim, hidden_size),  # 输入为潜在变量和状态拼接后的维度
            nn.Mish(),  # 激活函数增加非线性
            nn.Linear(hidden_size, hidden_size),  # 第二层线性层，维度保持隐藏大小
            nn.Mish(),  # 使用 Mish 激活
            nn.Linear(hidden_size, hidden_size),  # 第三层线性层，维度保持隐藏大小
            nn.Mish()  # 使用 Mish 激活函数
        )

        # 最终解码层，将隐藏层映射到动作空间维度
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_size, action_dim)  # 最后一层线性层，输出维度为动作空间维度
        )

        self.apply(init_weights)  # 初始化所有层的权重

        self.device = device  # 存储设备信息

    def encode(self, action, state):  # 编码器，将动作和状态编码为潜在变量
        x = torch.cat([action, state], dim=-1)  # 将动作和状态拼接在一起，生成输入
        result = self.encoder(x)  # 通过编码器的多层结构，得到编码结果
        result = torch.flatten(result, start_dim=1)  # 将结果展平，以便与均值和方差层对接

        mu = self.fc_mu(result)  # 通过均值层，获得潜在空间均值
        log_var = self.fc_var(result)  # 通过方差层，获得潜在空间对数方差

        return mu, log_var  # 返回均值和对数方差

    def decode(self, z, state):  # 解码器，将潜在变量和状态解码为动作
        x = torch.cat([z, state], dim=-1)  # 将潜在变量和状态拼接为解码器输入
        result = self.decoder(x)  # 通过解码器的多层结构，得到解码结果
        result = self.final_layer(result)  # 最后一层线性层，生成动作

        return result  # 返回解码后的动作

    def reparameterize(self, mu, logvar):  # reparameterization trick 生成潜在变量 z
        std = torch.exp(0.5 * logvar)  # 计算方差的平方根，即标准差
        eps = torch.randn_like(std)  # 生成与标准差相同形状的随机噪声
        return eps * std + mu  # 使用均值和标准差生成新的 z 样本

    def loss(self, action, state):  # 计算损失函数，包括重构损失和 KLD 损失
        mu, log_var = self.encode(action, state)  # 对动作和状态进行编码，得到均值和对数方差
        z = self.reparameterize(mu, log_var)  # 使用 reparameterization trick 得到潜在变量 z
        recons = self.decode(z, state)  # 对潜在变量 z 和状态进行解码，得到重构的动作

        kld_weight = 0.1  # KLD 损失的权重系数
        recons_loss = F.mse_loss(recons, action)  # 重构损失，计算重构的动作与原始动作的均方误差

        # KLD 损失，用于度量生成的高斯分布与标准正态分布的差异
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss  # 总损失是重构损失和加权 KLD 损失的和
        return loss  # 返回损失值

    def forward(self, state, eval=False):  # 前向传播，生成动作样本
        batch_size = state.shape[0]  # 获取批次大小
        shape = (batch_size, self.hidden_size)  # 定义潜在变量的形状

        if eval:  # 如果是评估模式
            z = torch.zeros(shape, device=self.device)  # 评估时用零向量代替 z
        else:  # 如果是训练模式
            z = torch.randn(shape, device=self.device)  # 随机生成潜在变量 z

        samples = self.decode(z, state)  # 对潜在变量 z 和状态进行解码，得到动作样本
        return samples.clamp(-1., 1.)  # 将动作限制在 [-1, 1] 的范围内并返回

    #限制动作范围----只能是返回某个范围内，不好具体控制车辆在哪个位置，返回哪个范围！