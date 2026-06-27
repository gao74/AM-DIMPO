# AM-DIMPO 项目说明

本仓库是论文 **AM-DIMPO: Action-Mask-Guided Safe Diffusion-Implicit Policy Optimization** 的实验代码整理版，对应论文链接：

[https://www.mdpi.com/2076-3417/16/8/3687](https://www.mdpi.com/2076-3417/16/8/3687)

项目面向高速公路匝道合流场景中的安全自动驾驶决策问题。整体思路是用扩散策略生成连续驾驶动作，再结合 DDIM 隐式采样、Critic 动作梯度优化和状态相关动作掩码，使智能体在追求高回报的同时尽量输出满足安全约束的动作。

> 仓库中与论文算法最直接对应的是 `am_dimpo_complete.py`。`dimpo_run.py`、`dipo_run.py` 和 `ppo_run.py` 更偏向实验脚本、早期版本或对比基线。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `am_dimpo_complete.py` | 按 AM-DIMPO 论文流程整理的完整注释版参考实现。 |
| `dimpo_run.py` | DIMPO / DDIM 版本的合流环境训练脚本。 |
| `dipo_run.py` | DiPo 风格扩散策略训练脚本。 |
| `ppo_run.py` | PPO 对比实验脚本。 |
| `agent/` | 策略网络、Critic、扩散模型、VAE、经验回放和辅助函数。 |
| `highway_env/` | 修改后的高速合流仿真环境，包括 `merge-v33`、`merge-v41`、`merge-v50` 等。 |
| `common/` | 绘图、日志、视频和通用工具代码。 |
| `d4rl/`, `dm_control/` | 实验中保留的环境与 benchmark 依赖代码。 |
| `requirements.txt` | 复现实验环境的 Python 依赖快照。 |

## 论文与代码对应关系

| 论文中的模块 / 思想 | 代码位置 | 对应说明 |
| --- | --- | --- |
| AM-DIMPO 总体训练框架 | `am_dimpo_complete.py` 的 `train_am_dimpo` | 构建环境、经验回放、AM-DIMPO 智能体、周期评估和最终 mask-on / mask-off 评估。 |
| 合流场景结构化状态表示 | `MergeStateExtractor`, `StructuredStateWrapper` | 将原始仿真状态整理为自车状态、道路几何、局部交通密度、可插入间隙和邻近车辆特征。 |
| 连续动作扩散策略 | `DiffusionActor`; 另见 `agent/diffusion.py`, `agent/diffusion_ddim.py` | 学习以状态和扩散时间步为条件的动作去噪模型。 |
| DDIM 隐式动作采样 | `sample_raw_action`, `get_ddim_timesteps`; 另见 `agent/diffusion_ddim.py` | 用较少采样步从噪声中生成连续控制动作，提高推理效率。 |
| 交通密度自适应采样步数 | `select_density_aware_steps` | 根据 100 m 范围内周围车辆数量切换 easy / hard 场景下的 DDIM 采样步数。 |
| 双 Q 网络 / Critic 估计 | `TwinCritic`; 另见 `agent/model.py` 的 `Critic` | 用两个 Q 网络评估动作价值，降低过估计并稳定离策略训练。 |
| Critic 引导的动作优化 | `refine_with_critic`; 另见 `agent/DiPo.py` 的 `action_gradient` | 对扩散模型采样出的动作沿 Q 值梯度方向做局部优化。 |
| 状态相关动作掩码 | `ActionMask` | 根据当前合流状态计算动作上下界，将原始动作裁剪到可行安全范围。 |
| 安全内化损失 | `AMDIMPOAgent.update` | 在扩散损失外加入 mask 修正量惩罚，使策略逐步学习直接生成更安全的动作。 |
| 离策略经验回放 | `ReplayBuffer`; 另见 `agent/replay_memory.py` | 存储交互轨迹和扩散动作样本，用于 Critic 更新和策略更新。 |
| 合流驾驶环境 | `highway_env/envs/merge_v33.py`, `merge_v41.py`, `merge_v50.py` | 实现车辆生成、奖励函数、终止条件、道路结构和不同交通密度变体。 |
| 对比实验 / 消融脚本 | `dipo_run.py`, `dimpo_run.py`, `ppo_run.py` | 用于扩散策略、DDIM 采样策略和 PPO 基线的实验对照。 |

## 核心实现说明

`am_dimpo_complete.py` 采用分段编号组织，适合按论文流程阅读：

1. `TrainConfig` 保存实验超参数，包括交通密度、扩散步数、DDIM 采样步数、动作范围和安全损失权重。
2. `MergeStateExtractor` 从合流环境中提取论文风格的结构化状态向量。
3. `DiffusionActor` 和 `DiffusionSchedule` 实现状态条件扩散策略。
4. `ActionMask` 定义连续、状态相关的可行动作集合。
5. `AMDIMPOAgent` 汇总扩散损失、Critic 损失、动作梯度优化、动作掩码和安全内化损失。
6. `train_am_dimpo` 负责训练循环、日志输出、周期评估和模型保存。

## 环境配置

依赖包记录在 `requirements.txt` 中。Linux / macOS 可参考：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell 可参考：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

MuJoCo、PyTorch、CUDA 等依赖可能需要根据本机环境单独配置。如果安装失败，建议先检查 Python 版本、CUDA/PyTorch 版本和 MuJoCo 运行时。

## 运行方式

运行注释版 AM-DIMPO 参考实现：

```bash
python am_dimpo_complete.py
```

运行 DDIM / DIMPO 实验脚本：

```bash
python dimpo_run.py --env_name merge-v33 --eta 0.02 --seed 0
```

运行 DiPo 风格扩散策略基线：

```bash
python dipo_run.py --env_name merge-v33 --seed 0
```

训练过程中产生的日志、模型权重、checkpoint 和视频会保存到各脚本配置的实验目录下。

## 建议阅读顺序

1. 先阅读论文，理解 AM-DIMPO 的扩散策略、安全动作掩码和合流场景实验设计。
2. 再阅读 `am_dimpo_complete.py`，按编号从配置、状态提取、扩散策略、动作掩码、智能体更新到训练循环逐段查看。
3. 对照 `agent/DiPo.py`、`agent/diffusion_ddim.py` 和 `dimpo_run.py`，理解模块化实验版本与完整参考实现的关系。
4. 最后查看 `highway_env/envs/merge_v33.py` 等环境文件，理解奖励函数、道路结构、车辆生成和终止条件。

## 说明

- 本仓库保留了较多实验依赖和环境代码，目的是尽量保存原始实验上下文。
- `am_dimpo_complete.py` 中对论文未完全展开的工程细节做了显式注释，例如具体几何常数、动作范围映射和 Critic 引导系数。
- 如果 README 与正式发表论文存在不一致，应以论文正文为准。
