import gymnasium as gym
import panda_gym
import os
import pathlib # 用于更安全地处理路径
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import RecordVideo

# --- 1. 路径自动定位 ---
# 获取当前脚本所在的文件夹
CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
# 向上寻找 logs 文件夹（假设 logs 在脚本同级或上一级）
# LOG_DIR = CURRENT_DIR / "logs" / "sac_panda_reach"
LOG_DIR = CURRENT_DIR / "outputs" / "experiment_0F" / "SAC_run_01_random_baseline" / "models"

# 尝试寻找模型文件（去掉 .zip 后缀，让 SB3 自己处理）
MODEL_PATH = str(LOG_DIR / "best_model") 
# 如果 best_model 不存在，尝试找 final 模型
if not os.path.exists(MODEL_PATH + ".zip"):
    MODEL_PATH = str(LOG_DIR / "sac_panda_reach_final")

STATS_PATH = str(LOG_DIR / "vec_normalize.pkl")

print(f"当前脚本目录: {CURRENT_DIR}")
print(f"预期的模型路径: {MODEL_PATH}.zip")

# --- 配置 ---
VIDEO_DIR = CURRENT_DIR / "outputs" / "experiment_0F" / "SAC_run_01_random_baseline" / "videos"


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