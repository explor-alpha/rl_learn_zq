"""
将 evaluate_simple 函数修改为以下版本，增加打印信息和强制步数限制，以排查问题：
"""
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from env import PlanarBringBallEnv
from config import TrainConfig


def evaluate_simple(model, env, n_episodes=5, max_steps_per_episode=500):
    """增加打印监控和强制超时保护的评估函数"""
    all_rewards = []
    print(f"开始评估，预计运行 {n_episodes} 个回合...")
    
    for i in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            orig_reward = env.get_original_reward()
            episode_reward += orig_reward[0]
            
            steps += 1
            # 强制超时保护：如果 500 步还不结束，强制跳出，防止死循环
            if steps > max_steps_per_episode:
                print(f"  警告：回合 {i} 达到最大步数 {max_steps_per_episode} 仍未结束，强制跳出。")
                break
                
        all_rewards.append(episode_reward)
        print(f"  回合 {i} 结束，步数: {steps}, 奖励: {episode_reward:.2f}")
        
    mean_r = np.mean(all_rewards)
    return mean_r

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