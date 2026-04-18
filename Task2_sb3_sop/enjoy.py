import gymnasium as gym
import panda_gym
import time
import os
import pathlib # 用于更安全地处理路径
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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

def enjoy():
    # 1. 创建环境
    env = gym.make("PandaReach-v3", render_mode="human")
    env = DummyVecEnv([lambda: env])

    # 2. 加载统计量
    if os.path.exists(STATS_PATH):
        print("成功加载归一化文件！")
        env = VecNormalize.load(STATS_PATH, env)
        env.training = False
        env.norm_reward = False
    else:
        print(f"❌ 错误：未找到归一化文件 {STATS_PATH}")
        return

    # 3. 加载模型
    try:
        model = SAC.load(MODEL_PATH)
        print("成功加载模型！")
    except Exception as e:
        print(f"❌ 无法加载模型，请检查文件是否存在。错误: {e}")
        return

    # 4. 运行
    obs = env.reset()
    ACTION_DT = 1 # 50Hz
    
    try:
        while True:
            start_time = time.time()
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env.step(action)
            
            if dones:
                obs = env.reset()
                time.sleep(0.05)
            
            elapsed = time.time() - start_time
            sleep_time = ACTION_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        env.close()

if __name__ == "__main__":
    enjoy()