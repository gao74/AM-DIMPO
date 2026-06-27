import argparse
import random
import sys
import time
import numpy as np
import torch
from gym.wrappers import RecordVideo
from agent.DiPo import DiPo
from agent.replay_memory import ReplayMemory, DiffusionMemory
from tensorboardX import SummaryWriter
import gym
import os
import datetime
import pickle

sys.path.append("../highway-env")
import highway_env


def readParser():
    parser = argparse.ArgumentParser(description='Diffusion Policy Training')

    # 环境参数
    parser.add_argument('--env_name', default="merge-v33", help='Environment name')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    # 训练参数
    parser.add_argument('--num_steps', type=int, default=1000000, help='Total environment steps')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Target smoothing coefficient')
    parser.add_argument('--start_steps', type=int, default=50000, help='Random action steps before training')

    # 策略参数
    parser.add_argument("--policy_type", type=str, default="Diffusion", help="Policy type")
    parser.add_argument("--beta_schedule", type=str, default="cosine", help="Beta schedule")
    parser.add_argument('--n_timesteps', type=int, default=100, help='Diffusion timesteps')
    parser.add_argument('--diffusion_lr', type=float, default=0.0003, help='Diffusion learning rate')
    parser.add_argument('--critic_lr', type=float, default=0.0003, help='Critic learning rate')
    parser.add_argument('--action_lr', type=float, default=0.03, help='Action learning rate')

    # 训练配置
    parser.add_argument('--max_steps_per_episode', type=int, default=150, help='Max steps per episode')
    parser.add_argument('--eval_interval', type=int, default=10000, help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=500, help='Model save interval')
    parser.add_argument('--video_interval', type=int, default=500, help='Video recording interval')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')

    # 路径参数
    parser.add_argument('--cuda', default='cuda:0', help='CUDA device')
    parser.add_argument('--base_dir', default='./experiments', help='Base directory for experiments')
    parser.add_argument('--experiment_name', default=None, help='Experiment name')
    parser.add_argument('--checkpoint_dir', default='./checkpoints', help='Checkpoint directory')

    # 恢复训练
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--resume_path', default=None, help='Specific checkpoint path to resume from')

    # 动作采样参数
    parser.add_argument('--noise_ratio', type=float, default=1.0, help='Noise ratio in sample process')
    parser.add_argument('--action_gradient_steps', type=int, default=20, help='Action gradient steps')
    parser.add_argument('--ratio', type=float, default=0.1, help='Ratio of action grad norm to action_dim')
    parser.add_argument('--ac_grad_norm', type=float, default=2.0, help='Actor and critic grad norm')
    parser.add_argument('--update_actor_target_every', type=int, default=1, help='Update actor target per iteration')

    # 启发式动作控制
    parser.add_argument('--use_heuristic', action='store_true', help='Use heuristic actions during random phase')
    parser.add_argument('--heuristic_x_min', type=float, default=317, help='Min x position for heading adjustment')
    parser.add_argument('--heuristic_x_max', type=float, default=323, help='Max x position for heading adjustment')

    return parser.parse_args()


def setup_directories(args):
    """创建实验目录结构"""
    if args.experiment_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"dipo_{args.env_name}_seed{args.seed}_{timestamp}"
    else:
        experiment_name = args.experiment_name

    base_path = os.path.join(args.base_dir, experiment_name)
    dirs = {
        'base': base_path,
        'logs': os.path.join(base_path, "logs"),
        'videos': os.path.join(base_path, "videos"),
        'models': os.path.join(base_path, "models"),
        'checkpoints': os.path.join(base_path, "checkpoints"),
        'training_logs': os.path.join(base_path, "training_logs")
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def save_checkpoint(agent, memory, diffusion_memory, steps, episodes, checkpoint_dir):
    """保存训练检查点"""
    checkpoint = {
        'steps': steps,
        'episodes': episodes,
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
        'random_state': random.getstate(),
        'np_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
        'memory': memory,
        'diffusion_memory': diffusion_memory,
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{steps}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"检查点已保存至 {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(agent, checkpoint_path):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    random.setstate(checkpoint['random_state'])
    np.random.set_state(checkpoint['np_random_state'])
    torch.set_rng_state(checkpoint['torch_random_state'])
    steps = checkpoint['steps']
    episodes = checkpoint['episodes']
    memory = checkpoint['memory']
    diffusion_memory = checkpoint['diffusion_memory']
    print(f"从检查点 {checkpoint_path} 恢复训练")
    return steps, episodes, memory, diffusion_memory


def find_latest_checkpoint(checkpoint_dir):
    """查找最新的检查点文件"""
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pth')]
    if not checkpoint_files:
        return None
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    return os.path.join(checkpoint_dir, latest_checkpoint)


def evaluate(env, agent, writer, steps):
    """评估模型性能"""
    episodes = 10
    returns = np.zeros((episodes,), dtype=np.float32)

    for i in range(episodes):
        state = env.reset()
        episode_reward = 0.
        done = False
        while not done:
            action = agent.sample_action(state, eval=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        returns[i] = episode_reward

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    writer.add_scalar('reward/test_mean', mean_return, steps)
    writer.add_scalar('reward/test_std', std_return, steps)

    print('-' * 60)
    print(f'评估步数: {steps:<6}  平均奖励: {mean_return:<6.1f} ± {std_return:<4.1f}')
    print('-' * 60)

    return mean_return


def heuristic_action(env, step_count, args):
    """启发式动作选择（基于原始代码的逻辑）"""
    action = env.action_space.sample()
    action = np.array(action)

    vehicle_attributes = vars(env.vehicle)
    position = vehicle_attributes['position']
    x_position = position[0]
    heading_value = vehicle_attributes['heading']
    lane_index = vehicle_attributes.get('lane_index', None)

    # 位置调整逻辑
    if args.heuristic_x_min < x_position < args.heuristic_x_max:
        action[1] = -heading_value
    else:
        if step_count == 0:
            action[1] = -0.12
        elif lane_index == ('k', 'b', 0):
            action[1] = 0
        elif lane_index == ('b', 'c', 2):
            action[1] = np.random.uniform(-0.15, 0.1)
        elif lane_index == ('b', 'c', 0):
            if heading_value > 0:
                action[1] = np.random.uniform(-0.15, 0)
            elif heading_value < 0:
                action[1] = np.random.uniform(0, 0.15)
            else:
                action[1] = 0
        else:
            action[1] = np.random.uniform(-0.15, 0.1)

    return action


def record_video(env, agent, video_dir, episode_num, num_episodes=3):
    """录制评估视频"""
    video_save_dir = os.path.join(video_dir, f"episode_{episode_num}")
    os.makedirs(video_save_dir, exist_ok=True)

    video_env = RecordVideo(env, video_folder=video_save_dir, episode_trigger=lambda x: True)

    for video_episode in range(num_episodes):
        state = video_env.reset()
        done = False

        while not done:
            action = agent.sample_action(state, eval=True)
            next_state, reward, done, _ = video_env.step(action)
            state = next_state

    video_env.close()
    print(f"已录制 {num_episodes} 回合视频，保存至: {video_save_dir}")


def save_training_info(info_dict, log_dir, filename="training_info.txt"):
    """保存训练信息"""
    info_path = os.path.join(log_dir, filename)
    with open(info_path, "w") as f:
        for key, value in info_dict.items():
            f.write(f"{key}: {value}\n")
    print(f"训练信息已保存至 {info_path}")


def main():
    args = readParser()

    # 设置设备
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建目录
    dirs = setup_directories(args)

    # 初始化TensorBoard
    writer = SummaryWriter(dirs['logs'])

    # 创建环境
    env = gym.make(args.env_name)
    state_size = int(np.prod(env.observation_space.shape))
    action_size = int(np.prod(env.action_space.shape))

    print(f"状态空间大小: {state_size}, 动作空间大小: {action_size}")

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    # 初始化经验池
    memory_size = int(1e6)
    memory = ReplayMemory(state_size, action_size, memory_size, device)
    diffusion_memory = DiffusionMemory(state_size, action_size, memory_size, device)

    # 初始化智能体
    agent = DiPo(args, state_size, env.action_space, memory, diffusion_memory, device)

    # 训练参数
    total_steps = 0
    total_episodes = 0
    updates_per_step = 1

    # 恢复训练
    if args.resume:
        if args.resume_path:
            checkpoint_path = args.resume_path
        else:
            checkpoint_path = find_latest_checkpoint(dirs['checkpoints'])

        if checkpoint_path and os.path.exists(checkpoint_path):
            total_steps, total_episodes, memory, diffusion_memory = load_checkpoint(agent, checkpoint_path)
            print(f"从步数 {total_steps}, 回合数 {total_episodes} 恢复训练")
        else:
            print("未找到检查点文件，从头开始训练")

    # 保存训练配置
    config_info = {
        "实验开始时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "环境名称": args.env_name,
        "随机种子": args.seed,
        "总步数": args.num_steps,
        "批次大小": args.batch_size,
        "折扣因子": args.gamma,
        "策略类型": args.policy_type,
        "扩散步数": args.n_timesteps,
        "学习率(扩散)": args.diffusion_lr,
        "学习率(评论家)": args.critic_lr,
        "学习率(动作)": args.action_lr,
        "开始训练步数": args.start_steps,
        "最大回合步数": args.max_steps_per_episode,
        "评估间隔": args.eval_interval,
        "使用启发式": args.use_heuristic,
        "设备": str(device)
    }
    save_training_info(config_info, dirs['base'], "experiment_config.txt")

    # 创建训练日志文件
    training_log_path = os.path.join(dirs['training_logs'], "training_log.txt")

    # 训练循环
    start_time = time.time()
    best_eval_reward = -float('inf')

    print(f"开始训练，目标步数: {args.num_steps}")

    with open(training_log_path, "a" if args.resume else "w") as log_file:
        while total_steps < args.num_steps:
            episode_start_time = time.time()
            episode_reward = 0.
            episode_steps = 0
            state = env.reset()
            done = False

            # 显示当前回合数
            print(f"开始第 {total_episodes + 1} 回合")

            while not done and episode_steps < args.max_steps_per_episode:
                # 动作选择
                if total_steps < args.start_steps:
                    if args.use_heuristic:
                        action = heuristic_action(env, episode_steps, args)
                    else:
                        action = env.action_space.sample()
                        action = np.array(action)
                else:
                    action = agent.sample_action(state, eval=False)

                # 环境交互
                next_state, reward, done, _ = env.step(action)
                mask = 0.0 if done else args.gamma

                # 记录日志
                log_file.write(
                    f"回合:{total_episodes + 1}, 步数:{episode_steps}, 动作:{action}, 奖励:{reward:.3f}, 结束:{done}\n")
                log_file.flush()  # 确保立即写入

                # 更新计数器
                total_steps += 1
                episode_steps += 1
                episode_reward += reward

                # 渲染环境
                env.render()

                # 存储经验
                agent.append_memory(state, action, reward, next_state, mask)

                # 开始训练
                if total_steps >= args.start_steps:
                    agent.train(updates_per_step, batch_size=args.batch_size, log_writer=writer)

                # 评估模型
                if total_steps % args.eval_interval == 0:
                    print("开始评估...")
                    eval_reward = evaluate(env, agent, writer, total_steps)

                    # 保存最佳模型
                    if eval_reward > best_eval_reward:
                        best_eval_reward = eval_reward
                        agent.save_model(dir=dirs['models'], id="best")
                        print(f"新的最佳模型! 奖励: {best_eval_reward:.2f}")

                    # 保存检查点
                    save_checkpoint(agent, memory, diffusion_memory, total_steps, total_episodes, dirs['checkpoints'])

                # 保存模型
                if total_episodes % args.save_interval == 0 and total_episodes > 0:
                    agent.save_model(dir=dirs['models'], id=total_episodes)
                    print(f"模型已保存至: {dirs['models']}/model_weights_ep{total_episodes}.pth")

                # 录制视频
                if total_episodes % args.video_interval == 0 and episode_steps == 1:
                    record_video(env, agent, dirs['videos'], total_episodes)

                state = next_state

            # 回合结束
            total_episodes += 1

            # 记录回合信息
            episode_time = time.time() - episode_start_time
            log_file.write(f"回合 {total_episodes} 在 {episode_time:.2f} 秒内完成，总奖励: {episode_reward:.2f}\n\n")

            # TensorBoard记录
            if total_episodes % args.log_interval == 0:
                writer.add_scalar('reward/train', episode_reward, total_steps)
                writer.add_scalar('episode/length', episode_steps, total_steps)
                writer.add_scalar('episode/time', episode_time, total_steps)

            print(
                f'回合: {total_episodes:<4}  回合步数: {episode_steps:<4}  奖励: {episode_reward:<6.1f}  总步数: {total_steps}')

    # 训练结束
    total_time = time.time() - start_time

    # 保存最终模型
    agent.save_model(dir=dirs['models'], id="final")

    # 最终评估
    print("进行最终评估...")
    final_eval_reward = evaluate(env, agent, writer, total_steps)

    # 保存训练完成信息
    completion_info = {
        "训练完成时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "总训练时间(秒)": f"{total_time:.2f}",
        "总回合数": total_episodes,
        "总步数": total_steps,
        "最终评估奖励": f"{final_eval_reward:.2f}",
        "最佳评估奖励": f"{best_eval_reward:.2f}",
        "平均每回合时间(秒)": f"{total_time / total_episodes:.2f}" if total_episodes > 0 else "N/A"
    }
    save_training_info(completion_info, dirs['base'], "training_completion.txt")

    print(f"训练完成!")
    print(f"总训练时间: {total_time:.2f} 秒")
    print(f"总回合数: {total_episodes}")
    print(f"总步数: {total_steps}")
    print(f"最终评估奖励: {final_eval_reward:.2f}")
    print(f"结果保存在: {dirs['base']}")

    env.close()
    writer.close()


if __name__ == "__main__":
    main()
'''
基本训练命令：
bash
# 基本训练
python dipo_run.py  --env_name merge-v33 --seed 0

# 使用启发式动作
python dipo_run.py  --use_heuristic

# 自定义实验名称
python dipo_run.py --experiment_name "my_experiment" --env_name merge-v6 --seed 42
高级配置：
bash
# 自定义训练参数
python dipo_unified.py \
    --env_name merge-v6 \
    --seed 42 \
    --num_steps 500000 \
    --start_steps 20000 \
    --batch_size 128 \
    --max_steps_per_episode 100 \
    --eval_interval 5000 \
    --use_heuristic \
    --base_dir "./my_experiments"
恢复训练：
bash
# 从最新检查点恢复
python dipo_run.py --resume --experiment_name "your_experiment_name"

# 从特定检查点恢复
python dipo_run.py --resume --resume_path "/path/to/checkpoint.pth"
输出结构
text
experiments/
└── dipo_merge-v6_seed0_20240304_143022/
    ├── logs/                    # TensorBoard日志
    ├── models/                  # 保存的模型权重
    ├── videos/                  # 录制的评估视频
    ├── checkpoints/             # 训练检查点
    ├── training_logs/           # 详细训练日志
    ├── experiment_config.txt    # 实验配置
    └── training_completion.txt  # 训练完成信息
主要特性
统一参数配置：整合了所有三个文件的参数

灵活的训练控制：支持步数控制和回合数控制

完整的日志系统：TensorBoard + 文本日志

智能恢复机制：可从任意检查点恢复训练

启发式动作支持：保留原始代码中的车辆控制逻辑

模块化设计：易于扩展和维护

这个整合脚本保留了所有原始文件的核心功能，同时提供了更好的可配置性和可维护性。
'''