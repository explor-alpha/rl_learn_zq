import gymnasium as gym
import panda_gym
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import RecordVideo

# --- 配置 ---
LOG_DIR = "./logs/sac_panda_reach/"
VIDEO_DIR = "./videos/"
STATS_PATH = os.path.join(LOG_DIR, "vec_normalize.pkl")
# 修改前：
# MODEL_PATH = os.path.join(LOG_DIR, "best_model.zip")
# 修改后：
MODEL_PATH = os.path.join(LOG_DIR, "best_model")

def record():
    # 1. 创建环境 (rgb_array 模式用于离线渲染视频)
    # 创建一个原始环境
    base_env = gym.make("PandaReach-v3", render_mode="rgb_array")
    
    # 使用 RecordVideo 包装器
    # episode_trigger 设定为 0 表示记录第一个 episode
    env = RecordVideo(
        base_env, 
        video_folder=VIDEO_DIR, 
        name_prefix="panda_reach_result",
        episode_trigger=lambda x: True # 记录所有回合
    )
    
    # 向量化
    env = DummyVecEnv([lambda: env])
    
    # 2. 加载归一化统计量
    if os.path.exists(STATS_PATH):
        env = VecNormalize.load(STATS_PATH, env)
        env.training = False
        env.norm_reward = False
    
    # 3. 加载模型
    model = SAC.load(MODEL_PATH)
    
    # 4. 运行几个回合
    print(f"正在录制视频并保存至 {VIDEO_DIR}...")
    for episode in range(3): # 录制 3 个完整回合
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            # 录制模式下不需要 time.sleep，让它以最快速度渲染
    
    env.close()
    print("录制完成！")

if __name__ == "__main__":
    record()