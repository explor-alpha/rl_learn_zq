"""
train.py: 自定义训练脚本
    对应任务: 平面机械手抓球——DeepMind Control Suite 中的 Manipulator 经典任务
    对应 MuJoCo 的模型实例: "manipulator_bring_ball.xml" 
    采用 stable_baselines3 框架

    支持功能：
    1. 课程学习
    2. 评估环境归一化同步
    3. 断点续训

--------------------------------------------------
Notes: 


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
--------------------------------------------------
"""
import time
import glob
import re
import os
# 应该是安装依赖的时候，conda pip 重复安装了 numpy，这里先跳过报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

from env import PlanarBringBallEnv
from config import TrainConfig

class SyncVecNormalizeCallback(BaseCallback):
    """
    Callback 回调函数： 确保同步训练环境与评估环境归一化统计数据

    PS: self.eval_env.ret_rms = self.train_env.ret_rms没必要。因为评估时不要对 reward 归一化
    对于 PPO, SyncVecNormalizeCallback 在 _on_rollout_end 触发。
    PPO 是先采集一个大的 Rollout(n_steps * n_envs)，然后才进行多次梯度更新

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
    EvalCallback: self.model.get_vec_normalize_env()=True 时，保存最优模型.zip
    SaveVecNormalizeCallback: 触发 self.model.get_vec_normalize_env()=True 时，同步保存对应的 .pkl

    Attributes:
        save_path: 统计数据(.pkl)的保存路径。
    """
    def __init__(self, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        """
        EvalCallback 在发现新最优模型时（即，触发 callback_on_new_best 时）
        会调用此处的 _on_step。
        """
        path = self.save_path
        if self.model.get_vec_normalize_env():
            self.model.get_vec_normalize_env().save(path)
        return True


class InfoLoggerCallback(BaseCallback):
    """
    自定义 Callback：从 env 的 info 中提取指标并记录到 TensorBoard
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.success_buffer = []

    def _on_step(self) -> bool:
        # self.locals['infos'] 是一个包含所有并行环境 info 字典的列表
        for info in self.locals['infos']:
            if "is_success" in info:
                self.success_buffer.append(info["is_success"])
                # 记录实时高度
                self.logger.record_mean("env/dist_ball_to_target", info["dist_b2t"])

        # 每 2048 步计算一次平均成功率并记录
        if len(self.success_buffer) >= 2048:
            self.logger.record("env/rolling_success_rate", np.mean(self.success_buffer))
            self.success_buffer = []
        return True


def course_evaluate(model, env, n_episodes=30):
    """
    使用 SB3 官方工具评估，自动处理 VecNormalize 和奖励统计
    """
    # 用于存储每个 episode 是否成功的列表
    successes = []

    def callback(local_vars, global_vars):
        """
        evaluate_policy 内部每一步都会调用这个闭包
        local_vars 包含了内部的 infos, dones, rewards 等
        """
        # 只有在 episode 结束时才记录成功标志
        # local_vars['dones'] 是一个布尔数组 (因为是 VecEnv)
        for i, done in enumerate(local_vars['dones']):
            if done:
                # 从 info 中提取是否成功
                info = local_vars['infos'][i]
                if "is_success" in info:
                    successes.append(info["is_success"])

    # evaluate_policy 会自动处理：
    # 1. 多个 episode 的循环
    # 2. VecNormalize 的原始奖励提取 (通过调用 env.get_original_reward())
    # 3. 确定性策略选择
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=n_episodes, 
        deterministic=True,
        callback=callback  # 注入我们的成功率统计逻辑
    )

    # 计算成功率
    success_rate = np.mean(successes) if successes else 0.0
    
    return mean_reward, success_rate

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
    # ------------ 1. cfg参数导入 & 基础配置 ------------ 
    cfg = TrainConfig()

    output_dir = cfg.output_dir
    best_dir = os.path.join(output_dir, "best")
    latest_dir = os.path.join(output_dir, "latest")
    stages_dir = os.path.join(output_dir, "stages")
    tb_log_dir = os.path.join(output_dir, "tb_logs")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(latest_dir, exist_ok=True)
    os.makedirs(stages_dir, exist_ok=True)

    # 初始化 stage_idx (默认从0开始)
    stage_idx = 0
    total_stages= len(cfg.curriculum_stages)

    # --- 断点续训 ---
    # 1.【choose：latest_dir 或 best_dir 或 stages_dir】
    find_dir = latest_dir  # 默认 latest_dir

    # 2. 尝试从 find_dir 寻找最新的模型和统计数据
    # 查找所有的 .zip 和 .pkl 文件
    model_files = glob.glob(os.path.join(find_dir, "*.zip"))
    stats_files = glob.glob(os.path.join(find_dir, "*.pkl"))

    # 初始化变量，防止 NameError！
    goon_model_path = None
    goon_stats_path = None

    if model_files and stats_files:
        # 按文件修改时间排序，取最新的一个
        goon_model_path = max(model_files, key=os.path.getmtime)
        goon_stats_path = max(stats_files, key=os.path.getmtime)
        print(f"检测到存档，开始断点续训")
        
        # 3. 从文件名中提取 stage_idx 
        try:
            match = re.search(r"stage-(\d+)", os.path.basename(goon_stats_path))
            if match:
                stage_idx = int(match.group(1))
                print(f"已恢复至课程阶段: {stage_idx}")
        except Exception:
            print("警告⚠️：未能从文件名提取 stage_idx, 将使用默认值 0")
    # --- 断点续训 ---

    # ------------ 2. train 环境配置 ------------ 
    """
    Monitor, 把每个 Episode 的真实奖励和长度写进 info 字典。仅用来画 TensorBoard 曲线。它不参与代码逻辑控制。
    DummyVecEnv, VecNormalize组合, 归一化输入
    VecNormalize: 归一化 input, 需配套 .pkl
    """
    def make_env():
        env = PlanarBringBallEnv(model_path=cfg.xml_path)
        env = Monitor(env)
        return env
    venv = DummyVecEnv([make_env])

    # 断点续训
    # 注意 None 导致的 TypeError
    if goon_model_path and os.path.exists(goon_model_path):
        env = VecNormalize.load(goon_stats_path, venv)
        model = PPO.load(goon_model_path, env=env)
    else:
        print(f"未检测到存档，开始全新训练...")
        env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.)
        model = PPO("MlpPolicy", env, 
                    learning_rate=cfg.learning_rate,
                    n_steps=cfg.n_steps,   # 每次训练前采集的总样本量 = n_steps * n_envs
                    # 减小 n_step：样本多样性下降，可能导致梯度估计不够准确（噪声大）。
                    # 减小 n_step：计算优势函数 GAE 时能回溯的时间步较短，对于需要长时记忆的任务可能不利。
                    batch_size=cfg.batch_size,
                    verbose=1,
                    tensorboard_log=tb_log_dir)

    # ------------ 3. eval 环境与 Callback 配置 (监控与最优保存)  ------------ 
    """
    eval 环境：
        独立于 train 环境
        training=False 表示不更新统计值
        norm_reward=False 不归一化奖励
        callback_after_eval=sync_callback 每次评估后同步归一化参数 (主要是 observation)
    保存：
        .npz
        best 模型（防止过拟合）
        自定义 info
    """
    eval_venv = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_venv, training=False, norm_reward=False)

    while stage_idx < total_stages:
        # --- best 保存路径(覆盖逻辑) ，仅最后阶段保存，减小计算开销---
        if_save_best = True  # if_save_best = (stage_idx == total_stages - 1)
        best_model_dir = None
        save_vec_callback = None

        if if_save_best:
            print(f"检测到当前为最后阶段 (Stage {stage_idx})，将开启最优模型记录...")
        
            b_total_steps = model.num_timesteps
            b_name = f"stage-{stage_idx}_step-{b_total_steps}"
            best_model_dir = best_dir
            b_stats_filename = f"vec-normalize_{b_name}.pkl"

            for file in os.listdir(best_dir):
                file_path = os.path.join(best_dir, file)
                try:
                    if os.path.isfile(file_path): os.unlink(file_path)
                except Exception as e: print(f"清理失败: {e}")

            best_stats_path = os.path.join(best_dir, b_stats_filename)
            save_vec_callback = SaveVecNormalizeCallback(save_path=best_stats_path)
        # ------------------------------

        # Callbacks
        info_callback = InfoLoggerCallback()
        sync_callback = SyncVecNormalizeCallback(env, eval_env)

        eval_callback = EvalCallback(
            eval_env, 
            best_model_save_path=best_model_dir,   # save best model(.zip)；EvalCallback中此参数应该是 zip 的上级文件夹
            log_path=output_dir, 
            eval_freq=cfg.eval_freq_steps, # 每四轮梯度回传参数更新评估一次
            deterministic=True, 
            render=False,
            callback_on_new_best=save_vec_callback, # 同步 save best 归一化参数(.pkl)
            callback_after_eval=sync_callback # 每次评估后再次强制同步
        )

        # ------------ 4. 课程学习 ------------ 
        """
        存档: 
            最新数据（覆盖逻辑）—用于断点续训：位于 outputs/latest/
            各课程阶段的数据（全保存）：位于 outputs/stages/
        晋级标准：
            通过 course_evaluate 函数,在评估环境测试 30 组 episode, 若成功率大于 80%，则晋级
        """
        stage = cfg.curriculum_stages[stage_idx]
        print(f"\n================ [阶段 {stage_idx}] 墙高: {stage['wall_height']} ================")
        
        # 1. 更新环境难度
        env.env_method("set_wall_height", stage['wall_height'])
        eval_env.env_method("set_wall_height", stage['wall_height'])

        # 2. 训练
        model.learn(
            total_timesteps=cfg.total_timesteps, 
            reset_num_timesteps=False,   # 保证 Tensorboard 曲线的连续性，不会在进入新课程时从 0 开始
            tb_log_name=f"stage_{stage_idx}",  
            callback=[eval_callback, info_callback]
        )

        # 3. 阶段结课评估
        mean_reward, success_rate = course_evaluate(model, eval_env, n_episodes=30)
        print(f"阶段 {stage_idx} 结课评估完成 | 成功率: {success_rate:.2%} | 平均奖励: {mean_reward:.2f}")

        # 保存文件名格式
        current_total_steps = model.num_timesteps
        timestamp = time.strftime("%H%M") 
        base_name = f"stage-{stage_idx}_step-{current_total_steps}_mean_reward-{mean_reward:.2f}_{timestamp}"
        model_filename = f"model_{base_name}.zip"
        stats_filename = f"vec-normalize_{base_name}.pkl"

        # --- 逻辑 1：更新 latest 文件夹 (覆盖逻辑) ---
        print(f"正在更新 latest 存档...")
        """
        # 清空 latest 文件夹中的旧文件（确保只有最新的）
        for file in os.listdir(latest_dir):
            file_path = os.path.join(latest_dir, file)
            try:
                if os.path.isfile(file_path): os.unlink(file_path)
            except Exception as e: print(f"清理失败: {e}")
        """

        # 保存最新模型和环境统计
        latest_model_path = os.path.join(latest_dir, model_filename)
        latest_stats_path = os.path.join(latest_dir, stats_filename)
        model.save(latest_model_path)
        env.save(latest_stats_path)

        # --- 逻辑 2：晋级判断与 stages 存档 ---
        if success_rate >= 0.8:
            print(f"🎉 成功率 {success_rate:.2%} 达标，保存至 stages 并晋级！")
            
            # 将 latest 中的文件拷贝到 stages 文件夹（或者直接 save 到这里）
            stage_save_path = os.path.join(stages_dir, model_filename)
            stage_stats_path = os.path.join(stages_dir, stats_filename)
            
            # 也可以直接再次调用 save，这样更保险
            model.save(stage_save_path)
            env.save(stage_stats_path)
            
            stage_idx += 1
        else:
            print(f"❌ 成功率未达标，将在本阶段继续训练。")
            # 可以在这里根据需要调整学习率，防止在同一关卡卡死
            # model.learning_rate = 1e-4 

if __name__ == "__main__":
    train()