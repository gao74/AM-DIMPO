import os
import sys
import time
import datetime
import gym
import numpy as np
from collections import deque
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import random
import matplotlib.pyplot as plt
from traci import vehicle
from highway_env.envs import MergeEnv
from highway_env.vehicle.kinematics import Vehicle
from gym.wrappers import RecordVideo

sys.path.append("../highway-env")
import highway_env
import torch
import pickle

# 检查是否可以使用GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 设置参数
total_timesteps = 60000
exploration_timesteps = 700  # 前50%为随机探索
training_timesteps = total_timesteps - exploration_timesteps  # 后50%为PPO训练
buffer_size = 100000  # 经验池大小
priority_alpha = 0.6  # 优先经验采样的alpha参数

random_number = random.randint(0, 15)

# 初始化环境
original_env = gym.make('merge-v35')  # 修改为 merge-v26
video_save_dir = os.path.join("D:/DIPO/ppo_train_new/training_videos")  # 修改为新的视频保存目录
os.makedirs(video_save_dir, exist_ok=True)
env = RecordVideo(original_env, video_folder=video_save_dir, episode_trigger=lambda x: False)  # 初始不自动录制
env.reset()

# 经验池和训练状态保存路径
experience_buffer_path = "experience_buffer_new0513.pkl"  # 修改为新的经验池文件
training_state_path = "training_state_new0513.pkl"  # 修改为新的训练状态文件
model_save_dir = "./ppo_merge_weights_train_new0512"  # 修改为新的模型保存目录
os.makedirs(model_save_dir, exist_ok=True)

# 加载或初始化经验池
if os.path.exists(experience_buffer_path):
    with open(experience_buffer_path, 'rb') as f:
        experience_buffer = pickle.load(f)
    print("已从文件加载经验池。")
    experience_buffer = deque(maxlen=buffer_size)
    print("初始化新的经验池。")
    # experience_buffer = deque(maxlen=buffer_size)
    # print("初始化新的经验池。")
else:
    experience_buffer = deque(maxlen=buffer_size)
    print("初始化新的经验池。")

# 加载或初始化训练状态
if os.path.exists(training_state_path):
    with open(training_state_path, 'rb') as f:
        training_state = pickle.load(f)
    start_episode = training_state['episode']
    episode_rewards = training_state['episode_rewards']
    episode_count = training_state['episode_count']
    step_count = training_state['step_count']
    print(f"从第 {start_episode} 回合恢复训练。")

    # start_episode = 0
    # episode_rewards = []
    # episode_count = 0
    # step_count = 0
    # print("初始化训练状态。")
else:
    start_episode = 0
    episode_rewards = []
    episode_count = 0
    step_count = 0
    print("初始化训练状态。")


# 定义经验池的优先采样
def sample_from_buffer(batch_size):
    if len(experience_buffer) == 0:
        print("经验池为空，无法采样。")
        return []
    priorities = [abs(exp[2]) ** priority_alpha for exp in experience_buffer]
    total_priority = sum(priorities)
    if total_priority > 0:
        probabilities = [p / total_priority for p in priorities]
    else:
        probabilities = [1 / len(experience_buffer)] * len(experience_buffer)
    indices = random.choices(range(len(experience_buffer)), weights=probabilities, k=batch_size)
    batch = [experience_buffer[i] for i in indices]
    return batch


# 创建文件保存训练日志（追加模式）
log_file = open("training_log_ppo_new.txt", "a")  # 修改为新的日志文件
log_file.write("探索和训练日志\n")
log_file.write("=====================================\n")

# 随机探索阶段
obs = env.reset()

for episode in range(start_episode, exploration_timesteps):
    adjust_direction = False
    abs_kb = False
    n1 = 0
    if episode % 300 == 0:
        temp_video_save_dir = os.path.join(video_save_dir, f"episode_{episode}")
        os.makedirs(temp_video_save_dir, exist_ok=True)
        temp_env = RecordVideo(original_env, video_folder=temp_video_save_dir, episode_trigger=lambda x: True)
        for _ in range(2):
            obs = temp_env.reset()
            done = False
            while not done:
                action = temp_env.action_space.sample()  # 随机选择动作
                action = np.array(action)
                next_obs, reward, done, _ = temp_env.step(action)
                obs = next_obs
        temp_env.close()

    for step in range(140):  # 每回合进行80步
        env.render()

        # 确保每回合的第一个动作固定为 [0.3, -0.12]
        if step == 0:
            action = np.array([0.12, -0.12])
        else:
            action = env.action_space.sample()  # 随机选择动作
            action = np.array(action)

        vehicle_attributes = vars(env.env.vehicle)
        lane_index = vehicle_attributes.get('lane_index', None)
        position = vehicle_attributes['position']
        x_position = position[0]
        heading_value = vehicle_attributes['heading']

        if x_position > 315 and x_position < 323 and not adjust_direction:
            action[1] = -heading_value
        else:
            if step == 0:
                action[1] = -0.14
            elif step >= 0 and lane_index == ('j', 'k', 0):
                action[1] = -heading_value
            elif lane_index == ('k', 'b', 0) and not abs_kb:
                action[1] = 0
                abs_kb = True

        next_obs, reward, done, _ = env.step(action)
        time.sleep(0)
        env.render()
        experience_buffer.append((obs, action, reward, next_obs, done))  # 存储经验

        log_file.write(f"探索回合: {episode + 1}, 步骤: {step + 1}, 动作: {action}, 奖励: {reward}\n")
        print(f"探索回合: {episode + 1}, 步骤: {step + 1}, 动作: {action}, 奖励: {reward}")

        obs = next_obs
        if done and step >= 120:
            obs = env.reset()
            break

    # 保存经验池和训练状态
    with open(experience_buffer_path, 'wb') as f:
        pickle.dump(experience_buffer, f)
    training_state = {
        'episode': episode + 1,
        'episode_rewards': episode_rewards,
        'episode_count': episode_count,
        'step_count': step_count
    }
    with open(training_state_path, 'wb') as f:
        pickle.dump(training_state, f)

# PPO训练阶段
checkpoint_callback = CheckpointCallback(save_freq=500, save_path=model_save_dir, name_prefix="ppo_merge")

# 加载或初始化PPO模型
final_model_path = os.path.join(model_save_dir, "ppo_merge_final.zip")
if os.path.exists(final_model_path):
    model = PPO.load(final_model_path, env=env, device=device)
    print("已从文件加载PPO模型。")
else:
    model = PPO("MlpPolicy", env, verbose=1, device=device)
    print("初始化新的PPO模型。")

# 在训练阶段开始前加载训练状态
if os.path.exists(training_state_path):
    with open(training_state_path, 'rb') as f:
        training_state = pickle.load(f)
    start_episode = training_state['episode']
    episode_rewards = training_state['episode_rewards']
    episode_count = training_state['episode_count']
    step_count = training_state['step_count']
    # 检查是否有已保存的回合权重文件，如果有则加载
    latest_episode_model = os.path.join(model_save_dir, f"ppo_merge_episode_{episode_count}.zip")
    if os.path.exists(latest_episode_model):
        model = PPO.load(latest_episode_model, env=env, device=device)
        print(f"从第 {episode_count} 回合的权重文件 {latest_episode_model} 恢复模型。")
    print(f"从第 {episode_count} 回合继续训练。")
else:
    start_episode = exploration_timesteps
    episode_rewards = []
    episode_count = 0
    step_count = 0
    print("初始化训练状态。")

obs = env.reset()
episode_reward = 0
with tqdm(total=training_timesteps, desc="训练进度", unit="step") as pbar:
    for timestep in range(training_timesteps):
        batch_size = 64
        batch = sample_from_buffer(batch_size)
        if batch:  # 检查 batch 是否为空
            obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)
            obs_batch = np.array(obs_batch)
            action_batch = np.array(action_batch)
            reward_batch = np.array(reward_batch)
            next_obs_batch = np.array(next_obs_batch)
            done_batch = np.array(done_batch)
            # 注意：当前代码未使用 batch 数据训练 PPO，可选择添加行为克隆逻辑
            # 例如：model.policy.train_on_batch(obs_batch, action_batch)

        model.learn(total_timesteps=1, reset_num_timesteps=False)

        # # 确保每回合的第一个动作固定为 [0.3, -0.12]
        # if step_count == 0:
        #     action = np.array([0.12, -0.12])
        #     print("第一个动作已经固定设置")
        #
        # if step_count == 1:
        #     action = np.array([0.1, 0.05])
        #     print("第2个动作已经固定设置")
        #
        # # 确保每回合的第一个动作固定为 [0.3, -0.12]
        # if step_count == 2:
        #     action = np.array([0.1, 0])
        #     print("第3个动作已经固定设置")
        # else:
        #     action, _ = model.predict(obs, deterministic=True)

        action, _ = model.predict(obs, deterministic=True)
        
        # 新增：检测并修正负加速度
        original_action = action.copy() 
        if action[0] < 0:
            original_action = action.copy() # 保存原始动作用于日志
            action[0] = 0.2
        print(f"检测到负加速度，已修正为0: 原始动作={original_action}, 修正后动作={action}")
        #log_file.write(f"动作修正: 原始={original_action}, 修正后={action}\n")

        next_obs, reward, done, _ = env.step(action)

        experience_buffer.append((obs, action, reward, next_obs, done))

        log_file.write(f"回合数: {episode_count + 1}, 训练步骤: {timestep + 1}, 动作: {action}, 奖励: {reward}\n")
        print(f"训练步骤: {timestep + 1}, 动作: {action}, 奖励: {reward}")

        obs = next_obs
        episode_reward += reward
        step_count += 1
        pbar.update(1)
        env.render()

        if done:
            env.render()
            episode_rewards.append(episode_reward)
            print(f"回合 {episode_count + 1} 结束，总奖励: {episode_reward}")
            obs = env.reset()
            episode_reward = 0
            episode_count += 1
            step_count = 0

            # 保存当前回合的模型权重
            save_path = os.path.join(model_save_dir, f"ppo_merge_episode_{episode_count}.zip")
            model.save(save_path)
            print(f"回合 {episode_count} 结束，权重文件已保存至: {save_path}")

            # 保存训练状态
            training_state = {
                'episode': start_episode,  # 这里保持探索阶段的episode不变
                'episode_rewards': episode_rewards,
                'episode_count': episode_count,
                'step_count': step_count
            }
            with open(training_state_path, 'wb') as f:
                pickle.dump(training_state, f)

            # 每100回合保存视频和额外的模型权重
            if episode_count % 1 == 0:
                extra_save_path = os.path.join("D:/DIPO/ppo_train_new",
                                               f"model_weights_ep{episode_count}.zip")  # 修改为新的额外保存路径
                model.save(extra_save_path)
                print(f"额外权重文件已保存至: {extra_save_path}")

                video_save_dir = os.path.join("D:/DIPO/ppo_train_new/videos", f"episode_{episode_count}")  # 修改为新的视频保存目录
                os.makedirs(video_save_dir, exist_ok=True)
                video_env = RecordVideo(env, video_folder=video_save_dir, episode_trigger=lambda x: True)
                for video_episode in range(4):
                    obs = video_env.reset()
                    done = False
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, done, _ = video_env.step(action)
                video_env.close()

        # 保存经验池
        with open(experience_buffer_path, 'wb') as f:
            pickle.dump(experience_buffer, f)

# 保存最终模型
model.save(final_model_path)
print("训练完成，模型和奖励已保存。")

log_file.write(f"训练完成，总回合数: {episode_count}\n")
log_file.close()
env.close()

print("训练日志已保存到 'training_log_ppo_new.txt' 文件中。")
