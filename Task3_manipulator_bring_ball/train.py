"""
包含课程学习、评估环境归一化同步及自动存档功能。
加入打印点
"""

"""
SB3-环境配置：
1. 基础配置：
    - 通常将 XML 导入的环境通过三层包装。一层是用于记录。一层是向量化，转化成数组运行的方式，一层是归一化。
    - Monitor():仅用于记录，不影响训练。放在 DummyVecEnv 内部，它可以捕捉到每个独立子环境的完整 Episode 结束信号，记录最原始的 r（reward）和 l（length）。SB3 的 Logger 会自动寻找 Monitor 产生的数据并显示在 TensorBoard 的 rollout/ep_rew_mean 中。
    - DummyVecEnv(): 向量化，把一个或多个普通环境包装起来，让它们可以像处理数组一样同时运行。
    - VecNormalize(): 归一化数组，一般处理 observation 和 reward。一般不对 action 进行处理。
    神经网络对输入数值非常敏感，将不同量级的物理量（如角度 0.1 和坐标 500）统一到均值 0、方差 1 附近，能防止梯度爆炸或消失。
    因为通常环境的定义就通过 box 空间，将输入连续动作限制在-1~1 之间，而Policy network 的输出头通常带一个 tanh 函数将输出限制在 -1~1 之间。
2. 断点续训: 通过 env = VecNormalize.load(latest_stats_path, venv) 加载.pkl归一化参数
3. 评估环境：
    1. 初始阶段: DummyVecEnv(BaseEnv)
    2. 初始阶段 obs 归一化，不处理 reward 归一化。 eval_env = VecNormalize(eval_venv, training=False, norm_reward=False) 
    3. 训练阶段-定义 SyncVecNormalizeCallback: 
        通过 EvalCallback 的 Hook: callback_after_eval传入这个 callback。
        确保每一次评估时评估环境拿到的 observation 归一化参数都是最新



SB3-Callback-记录 reward & 保存最优model.zip+同步的 .pkl
- EvalCallback：本身可以配合 tensorboard 记录每次评估时的 reward。
- EvalCallback + Hook-callback_on_new_best(定义SaveVecNormalizeCallback)：通过评估的 reward 保存最优model.zip+同步的 .pkl

课程学习 + SB3-训练逻辑：

PS：
    Q: EvalCallback本身就能处理原始的reward？
    - EvalCallback 内部调用的是 SB3 的 evaluate_policy
    - 随定义的环境是否（norm_reward=False）决定是否在 tensorboard 中记录归一化奖励。
    Q: 设置 eval_freq: 建议设为 n_steps 的整数倍。
    Q：Monitor 中记录的是过去的表现。且是训练表现，动作采样具有随机性。故不作为课程评估的方式。
    Q: 如何查看目前的训练进度，一个 episode 跑多少步，是不是能直接通过 tensorboard? 结合之前的 monitor 操作。


PS-tensorboard：
- eval/mean_reward 定义的环境 norm_reward=False 则曲线显示的就是原始物理分数    

PS：
定义 Agent 时候传入的 cfg.n_steps 超参数：每次训练前采集的总样本量 = n_steps * n_envs (例如 512)
EvalCallback中传入的 eval_freq=cfg.n_steps*4, # 每四轮梯度回传参数更新评估一次
在课程学习定义逻辑当中 learn 函数里的 cfg.total_timesteps 表示每这么多步进行结课评估，用于判断是否进入下一课程。
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
    Callback 回调函数： 确保同步训练环境与评估环境归一化统计数据

    PS: self.eval_env.ret_rms = self.train_env.ret_rms没必要。因为评估时不要对 reward 归一化

    Attributes:
        train_env: 训练环境，包含最新的 obs_rms 和 ret_rms。
        eval_env: 评估环境，其统计数据将被更新。
    """
    def __init__(self, train_env: VecNormalize, eval_env: VecNormalize, verbose=0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        """每步执行的逻辑。只在 Rollout 结束或评估前同步，减少每步操作开销"""
        return True

    def _on_rollout_end(self) -> None:
        """在每一轮收集结束后同步统计数据。"""
        self.eval_env.obs_rms = self.train_env.obs_rms
        # self.eval_env.ret_rms = self.train_env.ret_rms # 评估时不要对 reward 归一化
        if self.verbose > 0:
            print("[Callback] 已同步 VecNormalize 统计数据到评估环境")

class SaveVecNormalizeCallback(BaseCallback):
    """
    SB3-Callback: 配合 EvalCallback, 在其保存最优模型时同步保存对应的 .pkl
    触发条件： EvalCallback 触发 callback_on_new_best 的同时
    设计思想：.pkl 是对环境归一化处理之后对应的 VecNormalize 参数，需同步保存，以便还原 reward 等等的真实值

    Attributes:
        save_path: 统计数据（.pkl）的保存目录。
    """
    def __init__(self, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        """
        EvalCallback 在发现新最优模型时（即，触发 callback_on_new_best 时）
        会调用此处的 _on_step。
        """
        path = os.path.join(self.save_path, "best_vec_normalize.pkl")
        if self.model.get_vec_normalize_env():
            self.model.get_vec_normalize_env().save(path)
            if self.verbose > 0:
                print(f"[Callback] 检测到新最优模型，统计数据已保存至: {path}")
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
    """
    1. cfg参数导入
        包括：
        - 路径配置：output；断点续训: latest_model；best_model等等
    2. 训练env+agent配置
        包括：
        - 环境配置：Monitor；DummyVecEnv；VecNormalize组合；
        - 环境配置包含断点续训参数导入；
        - agent 定义

    """
    # 1. cfg参数导入
    cfg = TrainConfig()

    output_dir = cfg.output_dir
    latest_model_path = cfg.latest_model_path
    latest_stats_path = cfg.latest_stats_path
    best_model_path = cfg.best_model_path

    # 2. 环境配置
    # Monitor，把每个 Episode 的真实奖励和长度写进 info 字典。仅用来画 TensorBoard 曲线。它不参与代码逻辑控制。
    # DummyVecEnv, VecNormalize组合，归一化输入
    # VecNormalize：归一化 input，需配套 .pkl
    def make_env():
        env = PlanarBringBallEnv(model_path=cfg.xml_path)
        env = Monitor(env) # <--- 核心修改：在这一层加上 Monitor
        return env
    venv = DummyVecEnv([make_env])
    print("DEBUG: DummyVecEnv 创建成功")

    # 断点续训
    if os.path.exists(latest_model_path) and os.path.exists(latest_stats_path):
        print(f"检测到 latest_model 存档，开始断点续训...")
        # 获取.pkl 恢复环境统计数据
        env = VecNormalize.load(latest_stats_path, venv)
        # 获取模型参数.zip，恢复模型
        model = PPO.load(latest_model_path, env=env)
    else:
        print(f"未检测到 latest_model 存档，开始全新训练...")
        env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.)
        model = PPO("MlpPolicy", env, 
                    learning_rate=cfg.learning_rate,
                    n_steps=cfg.n_steps,   # 每次训练前采集的总样本量 = n_steps * n_envs
                    # 减小 n_step：样本多样性下降，可能导致梯度估计不够准确（噪声大）。
                    # 减小 n_step：计算优势函数 GAE 时能回溯的时间步较短，对于需要长时记忆的任务可能不利。
                    batch_size=cfg.batch_size,
                    verbose=1,
                    tensorboard_log=output_dir)

    # 3. 评估环境与 Callback 配置 (监控与最优保存) 
    # 创建专门用于评估的独立环境
    eval_venv = DummyVecEnv([make_env])
    # 评估环境的归一化必须跟训练环境同步，但 training=False 表示不更新统计值
    eval_env = VecNormalize(eval_venv, training=False, norm_reward=False)
    print("DEBUG: eval_env 成功...")

    # 结合自定义 Callback 以便同步保存 pkl
    save_vec_callback = SaveVecNormalizeCallback(save_path=output_dir)
    sync_callback = SyncVecNormalizeCallback(env, eval_env)
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=best_model_path,
        log_path=output_dir, 
        eval_freq=cfg.eval_freq_steps, # 每四轮梯度回传参数更新评估一次
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
        
        # 1. 更新环境难度
        env.env_method("set_wall_height", stage['wall_height'], indices=0)
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
            total_timesteps=cfg.total_timesteps, 
            reset_num_timesteps=False,
            tb_log_name="PPO_training",  # tb_log_name=f"stage_{stage_idx}",
            callback=eval_callback
        )

        # 阶段性强制存档（用于断点续训）
        model.save(latest_model_path)
        env.save(latest_stats_path)

        # 3. 调用 evaluate 函数，获取真实奖励，并和阈值对比，以判断是否进入下一阶段
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