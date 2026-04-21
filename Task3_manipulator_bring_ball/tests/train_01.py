"""
1. evaluate 函数的安全围栏：
添加了 max_steps=500。即使模型在某些阶段表现极差（比如一直原地转圈不结束），它也会在 500 步后强制终止并返回奖励。
使用 env.get_original_reward()[0]。这保证了返回的奖励是 MuJoCo 原始定义的数值（如 100.0），而不是被缩放后的 0.001。
2. SaveVecNormalizeCallback 修正：
移除 _on_event，将保存逻辑移至 _on_step。当 EvalCallback 发现新高奖励时，它会主动调用这个 on_step。
3. SyncVecNormalizeCallback 增加了 _on_rollout_end 钩子，比每一步都同步更高效。
4. TensorBoard 连续性：
将 tb_log_name 固定为 "PPO_training"。这样所有的课程阶段都会记录在同一个 TensorBoard 序列中，方便观察整体趋势，而不是产生一堆 stage_0, stage_1 文件夹。
同步逻辑优化：
"""

import os
# 应该是安装依赖的时候，conda pip 重复安装了 numpy，这里先跳过报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

from env import PlanarBringBallEnv
from config import TrainConfig

class SyncVecNormalizeCallback(BaseCallback):
    """
    将同步统计数据的逻辑写成一个 Callback
    确保训练环境和评估环境的归一化统计数据同步。"""
    def __init__(self, train_env: VecNormalize, eval_env: VecNormalize, verbose=0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        # 只在 Rollout 结束或评估前同步，减少每步操作开销
        return True

    def _on_rollout_end(self) -> None:
        self.eval_env.obs_rms = self.train_env.obs_rms
        self.eval_env.ret_rms = self.train_env.ret_rms
    

class SaveVecNormalizeCallback(BaseCallback):
    """当 EvalCallback 发现最佳模型时，保存对应的 VecNormalize 统计数据。"""
    def __init__(self, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        # EvalCallback 触发 callback_on_new_best 时会调用此处
        path = os.path.join(self.save_path, "best_vec_normalize.pkl")
        if self.model.get_vec_normalize_env():
            self.model.get_vec_normalize_env().save(path)
            if self.verbose > 0:
                print(f"已保存最佳归一化统计数据到 {path}")
        return True


def evaluate(model, env, n_episodes=5, max_steps=500):
    """
    针对 VecNormalize 环境的鲁棒评估函数。
    1. 使用 get_original_reward 获取真实奖励。
    2. 增加 max_steps 防止环境死循环。
    """
    old_training = env.training
    old_norm_reward = env.norm_reward
    env.training = False
    env.norm_reward = False 

    all_rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            # 关键：获取未缩放的原始奖励
            episode_reward += env.get_original_reward()[0]
            steps += 1
        all_rewards.append(episode_reward)

    env.training = old_training
    env.norm_reward = old_norm_reward
    return np.mean(all_rewards)


def train():
    cfg = TrainConfig()

    # 1. cfg路径导入
    # output
    output_dir = cfg.output_dir
    # 断点续训: latest_model
    latest_model_path = cfg.latest_model_path
    latest_stats_path = cfg.latest_stats_path
    # best_model
    best_model_path = cfg.best_model_path

    # 2. 环境配置
    # DummyVecEnv, VecNormalize组合，归一化输入
    def make_env():
        env = PlanarBringBallEnv(model_path=cfg.xml_path)
        env = Monitor(env) # <--- 核心修改：在这一层加上 Monitor
        return env
    venv = DummyVecEnv([make_env])

    # 断点续训
    if os.path.exists(latest_model_path) and os.path.exists(latest_stats_path):
        print(f"检测到 latest_model 存档，开始断点续训...")
        # 恢复环境统计数据
        env = VecNormalize.load(latest_stats_path, venv)
        # 恢复模型
        model = PPO.load(latest_model_path, env=env)
    else:
        print(f"未检测到 latest_model 存档，开始全新训练...")
        env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.)
        model = PPO("MlpPolicy", env, 
                    learning_rate=cfg.learning_rate,
                    n_steps=cfg.n_steps,
                    batch_size=cfg.batch_size,
                    verbose=1,
                    tensorboard_log=output_dir)

    # 3. 评估环境与 Callback 配置 (监控与最优保存) 
    # 创建专门用于评估的独立环境
    eval_venv = DummyVecEnv([make_env])
    # 评估环境的归一化必须跟训练环境同步，但 training=False 表示不更新统计值
    eval_env = VecNormalize(eval_venv, training=False, norm_reward=False)
    
    # 结合自定义 Callback 以便同步保存 pkl
    sync_callback = SyncVecNormalizeCallback(env, eval_env)
    save_vec_callback = SaveVecNormalizeCallback(save_path=output_dir)

    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=best_model_path,
        log_path=output_dir, 
        eval_freq=2000, # 每 2000 步评估一次
        deterministic=True, 
        render=False,
        callback_on_new_best=save_vec_callback, # 发现新高时触发 pkl 保存
        callback_after_eval=sync_callback # 每次评估后再次强制同步
    )

    # 4. 课程学习
    stage_idx = 0
    # 关键：手动配置 logger，防止 _logger 缺失
    #model.set_logger(configure(output_dir, ["stdout", "tensorboard"]))

    while stage_idx < len(cfg.curriculum_stages):
        stage = cfg.curriculum_stages[stage_idx]
        print(f"\n>>> 进入课程阶段 {stage_idx}: 墙高 = {stage['wall_height']}")
        
        # 1.1. 更新环境难度
        env.env_method("set_wall_height", stage['wall_height'], indices=0)
        # 1.2. 更新评估环境的难度 (eval_env 也是 VecNormalize)
        eval_env.env_method("set_wall_height", stage['wall_height'], indices=0)

        """
        # --- 新增：在 TensorBoard 中记录当前阶段信息 ---
        # 记录阶段索引
        model.logger.record("curriculum/stage_idx", stage_idx)
        # 记录当前阶段的难度物理量
        model.logger.record("curriculum/wall_height", stage['wall_height'])
        # 立即写入一次日志
        model.logger.dump(step=model.num_timesteps)
        # -------------------------------------------
        """

        # 2. 训练一段时间
        # 训练
        model.learn(
            total_timesteps=50000, 
            reset_num_timesteps=False,
            tb_log_name="PPO_training",  # tb_log_name=f"stage_{stage_idx}",
            callback=eval_callback
        )

        # 阶段性强制存档（用于断点续训）
        model.save(latest_model_path)
        env.save(latest_stats_path)

        # 3. 简单评估
        mean_reward = evaluate(model, eval_env)
        print(f"阶段评估结束，平均奖励: {mean_reward:.2f} (目标: {stage['threshold']})")
        
        # 4. 判断是否晋级
        if mean_reward > stage['threshold']:
            print("--- 达标，进入下一课程！ ---")
            stage_idx += 1
        else:
            print("--- 课程未达标，继续练！ ---")



if __name__ == "__main__":
    train()