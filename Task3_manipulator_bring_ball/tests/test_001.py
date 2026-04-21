"""
n_steps=512：PPO 内部缓冲区变小。这意味着程序每采集 512 个动作就会进行一次计算并打印那张包含 fps、loss 的表格。如果控制台连这张表都没出，说明你的 env.step 函数内部卡死了。
total_timesteps=10000：分段训练。每 1 万步程序就会跳出 model.learn 执行一次 model.save。即使你中途强行关掉，你也能得到一个最近的 .zip 文件。
os.makedirs：脚本运行的第一秒就会在硬盘创建文件夹。如果创建失败，脚本会立刻崩溃报错，而不是闷头跑 3 小时。
get_original_reward()：解决了“评估永远是 0.00x”的问题，确保 mean_reward >= threshold 逻辑能跑通，让课程能往下进行。
"""


import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from env import PlanarBringBallEnv
from config import TrainConfig

def evaluate_simple(model, env, n_episodes=5):
    """最简化的评估：强制获取原始奖励"""
    all_rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            # 关键：从 VecNormalize 中提取未缩放的真实奖励
            orig_reward = env.get_original_reward()
            episode_reward += orig_reward[0]
            if done[0]: break
        all_rewards.append(episode_reward)
    return np.mean(all_rewards)

def train_minimal():
    cfg = TrainConfig()
    
    # 1. 确保目录存在（非常重要！）
    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"日志和模型将保存至: {os.path.abspath(cfg.output_dir)}")

    # 2. 创建训练环境
    def make_env():
        env = PlanarBringBallEnv(model_path=cfg.xml_path)
        return Monitor(env) # Monitor 必须在 VecNormalize 内部
    
    venv = DummyVecEnv([make_env])
    # 初始化归一化环境
    env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. 初始化模型
    # 注意：n_steps 设小一点（如 512），这样每 512 步就会触发一次日志写入
    model = PPO(
        "MlpPolicy", 
        env, 
        n_steps=512, 
        verbose=1, 
        tensorboard_log=cfg.output_dir
    )

    # 4. 课程学习循环
    stage_idx = 0
    while stage_idx < len(cfg.curriculum_stages):
        stage = cfg.curriculum_stages[stage_idx]
        wall_h = stage['wall_height']
        threshold = stage['threshold']
        
        print(f"\n>>>>>> 当前阶段: {stage_idx} | 墙高: {wall_h} | 目标奖励: {threshold}")
        
        # 修改环境参数
        env.env_method("set_wall_height", wall_h)

        # 训练：每轮训练 10,000 步就强制停下来评估一次
        model.learn(
            total_timesteps=10000, 
            reset_num_timesteps=False, 
            tb_log_name="ppo_curriculum"
        )

        # 保存最新的模型和统计数据
        model.save(os.path.join(cfg.output_dir, "latest_model"))
        env.save(os.path.join(cfg.output_dir, "latest_stats.pkl"))

        # 评估
        mean_reward = evaluate_simple(model, env)
        print(f"阶段评估结果: {mean_reward:.2f} / 目标: {threshold}")

        if mean_reward >= threshold:
            print("!!! 达标，进入下一阶段 !!!")
            stage_idx += 1
            # 达成目标时额外保存一个备份
            model.save(os.path.join(cfg.output_dir, f"stage_{stage_idx}_success"))
        else:
            print("--- 未达标，继续在本阶段磨练 ---")

if __name__ == "__main__":
    train_minimal()