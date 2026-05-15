"""train.py: 自定义训练脚本; 对应“平面-机械手：操控物体-跨障-送至目标位置”任务

    核心功能：
    1. 实验结果的结构化保存和快速加载：

        1.1 结构化保存 (train.py):
            outputs/v6.1_exp-02_PPO/
            ├── latest/           # [断点续训和代码调试] 结课周期(cfg.total_timesteps)结束时，所有模型
            │   ├── model_stage-3_step-4423680_mean_reward-912.37_1622.zip
            │   └── vec-normalize_stage-3_step-4423680_mean_reward-912.37_1622.pkl
            ├── stages/           # [阶段晋级里程碑] 结课周期(cfg.total_timesteps)结束时，由评估体系 B 评估的达标模型
            ├── best/             # [防崩溃] 训练时，由评估体系 A (高频评估) 评估的 reward 最高模型，防止过拟合/灾难性遗忘崩溃；仅在最后一个课程阶段开启
            └── tb_logs/          # TensorBoard

        1.2 快速加载 (show.py):
            - 最简化示例: `mjpython Task3_manipulator_bring_ball/show.py --exp_name "v6.1_exp-01_PPO" --choose_model "stages" --match_id stage-3`
                - `--exp_name` 参数: 指定了实验id (必须手动传入)
                - `--choose_model` 参数：选择模型类型；[best,latest,stages] 
                - `--match_id` 参数：通过文件名中的特定标识快速加载模型; 如步数'4423680'或奖励'912.37'；如果不指定，则默认选择 reward 最高的模型

    2. 评估系统:

        2.1 评估体系 A (基于 SB3-Callbacks 的高频监控): 
            - 评估指标(基于日志和 TensorBoard):
                TensorBoard or 日志/
                ├── [自定义监控] (来自 `train_env`-`info`-`InfoLoggerCallback`)
                │   ├── `env_episode/` : Episode-level, 监控 `rolling_success_rate`; `rolling_final_dist`
                │   ├── `env_step/`    : Step-level, 监控 `is_grasped`; `wall_h`
                │   └── `reward_parts/`: Step-level, 监控 11 个分项 reward, 如`11_reach`; `+_progress`; `total`
                └── [SB3 原生监控] 
                    ├── `eval/`        : (来自 `eval_env`-`EvalCallback`) 独立评估环境，确定性策略下的 `mean_ep_length`; `mean_reward` (原始物理总分); `success_rate` 等。
                    ├── `rollout/`     : (来自 `train_env`-`Monitor`) 平时练习表现。记录带有随机探索的 `ep_rew_mean`、`ep_len_mean`。
                    ├── `time/`        : (来自 `train_env`-PPO 本体) 系统运行效率监控，如 `fps`、`total_timesteps`。
                    └── `train/`       : (来自 `train_env`-PPO 本体) 神经网络训练参数监控，如 Actor/Critic 的 Loss、学习率等。

            - 数据采集（主要分为两部分）：
                1. 评估环境（稳定单环境；确定性策略）
                    - EvalCallback + Hook-callback_on_new_best(定义SaveVecNormalizeCallback): 同步保存 .pkl
                2. 训练环境（用于 debug, 优化奖励塑形）
                    - InfoLoggerCallback: 记录 TensorBoard/自定义模块

        2.2 评估体系 B (基于确定性策略的低频结课评估；基于`course_evaluate`;独立进行):
            - 评估指标：
                30 个 episode 在独立评估环境下评估 mean_reward, success_rate
                当 success_rate > 80% 时认定完成课程
            - 数据采集：
                每个课程周期(cfg.total_timesteps)结束时评估 1 次，用于判断是否进入下一课程

    3. 课程学习 (Curriculum Learning，通过动态调整障碍墙高度实现渐进式训练)
    4. 断点续训 (自动寻址并恢复环境归一化状态与课程学习进度)。
    5. 并行环境收集 PPO 数据
--------------------------------------------------
KeyNotes_核心逻辑实现: 

1. 断点续训
    - "断点续训"寻址;【choose: latest_dir 或 best_dir 或 stages_dir】; 默认 latest_dir
    - 加载对应 .zip & .pkl
    - 加载课程学习阶段

2. 环境包装 (基于 SB3):
    - XML —> PlanarBringBallEnv+随机种子 —> Monitor() —> DummyVecEnv() or SubprocVecEnv() —> VecNormalize()
        - PlanarBringBallEnv: 自定义
        - Monitor():仅用于记录，不影响训练。放在 DummyVecEnv 内部，它可以捕捉到每个独立子环境的完整 Episode 结束信号，记录最原始的 r（reward）和 l（length）。SB3 的 Logger 会自动寻找 Monitor 产生的数据并显示在 TensorBoard 的 rollout/ep_rew_mean 中。
        - DummyVecEnv(): 向量化，把一个或多个普通环境包装起来，让它们可以像处理数组一样同时运行。
        - SubprocVecEnv(): 向量化+并行环境；注意给不同环境设置不同随机种子
        - VecNormalize(): 归一化数组，一般处理 observation 和 reward。一般不对 action 进行处理。
    - 归一化处理：
        神经网络对输入数值非常敏感，将不同量级的物理量（如角度 0.1 和坐标 500）统一到均值 0、方差 1 附近，能防止梯度爆炸或消失。
        观测空间：
        动作空间: 不处理。Policy network 的输出头通常带一个 tanh 函数将输出限制在 -1~1 之间。
        奖励: 不处理。env.py 已经把奖励塑形在 [0,1] 左右
    -  Details: 
        - 调用底层环境的方法

3. 独立双环境：
    - 训练环境：并行；随机策略；观测归一化处理/保存/同步
    - 评估环境: 单环境；确定性策略；观测归一化处理/保存/同步；
        - `DummyVecEnv()`
        - `eval_env = VecNormalize(eval_venv, training=False, norm_reward=False)`
        - callbacks

4. Details:
    - EvalCallback 内部调用的是 SB3 的 evaluate_policy; tensorboard 中 eval/mean_reward 定义的环境 norm_reward=False 则曲线显示的就是原始物理分数
    - Monitor 中记录的是过去的表现。且是训练表现，动作采样具有随机性。故不作为课程评估的方式。
    - 评估环境是否需要套 Monitor 层?
    - eval_freq 监控的是单环境调用 env.step() 的次数！故无需 * n_envs
    - 1次 rollout = PPO 采集 buffer 的阶段 (即收集 n_steps * n_envs 条 experience 的阶段)；例如，SB3 中：
        _on_rollout_end：这是 Callback 中的一个钩子函数。它意味着“这一大批数据刚刚收集完，即将拿去算梯度更新参数了”。（我们在 SyncVecNormalizeCallback 中把同步放在这里，就是为了在这个最稳定的时间点更新观测均值）。
        TensorBoard 中的 rollout/ep_rew_mean：表示在这一轮实战演习（Rollout）期间，那些恰好走到结局（Done）的回合，它们的平均总得分是多少。
        TensorBoard 中的 rollout/ep_len_mean：表示在这一轮演习中，平均一个回合能存活多少步。
--------------------------------------------------
"""
import time
import glob
import re
import os
# 应该是安装依赖的时候，conda pip 重复安装了 numpy，这里先跳过报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from collections import deque
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO

from env import PlanarBringBallEnv
from config import TrainConfig


class SaveVecNormalizeCallback(BaseCallback):
    """保存观测归一化参数的回调函数。
    EvalCallback: self.model.get_vec_normalize_env()=True 时，保存最优模型.zip
    SaveVecNormalizeCallback: 触发 self.model.get_vec_normalize_env()=True 时，同步保存对应的 .pkl
    通过`callback_on_new_best`调用
    确保推理或断点续训时能完全还原当时的观测分布。

    Attributes:
        save_path: 统计数据(.pkl)的保存路径。
    """
    def __init__(self, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        path = self.save_path
        if self.model.get_vec_normalize_env():
            self.model.get_vec_normalize_env().save(path)
        return True


class InfoLoggerCallback(BaseCallback):
    """自定义 TensorBoard 监控记录器。

    从 Env 抛出的 `info` 字典中提取物理状态与奖励分项，

    TensorBoard 新增目录：
        reward_parts/  # Step-level, 监控各分项 reward
        env_step/  # Step-level, 监控 `is_grasped`; `wall_h`
        env_episode/  # Episode-level, 监控 `rolling_success_rate`; `rolling_final_dist`
    """
    def __init__(self, window_size=100, verbose=0):
        super().__init__(verbose)
        self.success_buffer = deque(maxlen=window_size)
        self.final_dist_buffer = deque(maxlen=window_size)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")  # self.locals['dones'] 包含当前这一个 step 中，哪些并行环境结束了 Episode
        infos = self.locals.get("infos")

        for idx, info in enumerate(infos):
            
            # --- A. Step-level, 每 step 后记录的数据 ---
            if "reward_components" in info:
                for key, val in info["reward_components"].items():
                    self.logger.record_mean(f"reward_parts/{key}", val)  # 拆解奖励项打入 TensorBoard (得益于你的命名，Tensorboard 会自动有序排列)
            if "is_grasped" in info:
                self.logger.record_mean("env_step/grasp_maintain_rate", info["is_grasped"])  # 核心状态监控
            if "wall_h" in info:
                self.logger.record_mean("env_step/current_wall_height", info["wall_h"])

            # --- B1. Episode-level 仅在回合结束 (Done) 时记录的数据，反映最终结果 ---
            if dones[idx]:
                if "is_success" in info:
                    self.success_buffer.append(info["is_success"])
                if "dist_b2t" in info:
                    self.final_dist_buffer.append(info["dist_b2t"])  # 回合结束瞬间，球距离目标的最终物理距离

        # --- B2. 将滑动窗口平均值写入 Logger ---
        if len(self.success_buffer) > 0:
            self.logger.record("env_episode/rolling_success_rate", np.mean(self.success_buffer))
        if len(self.final_dist_buffer) > 0:
            self.logger.record("env_episode/rolling_final_dist", np.mean(self.final_dist_buffer))

        return True


def course_evaluate(model, env, n_episodes=30):
    """执行当前课程阶段的评估。

    使用确定的策略在评估环境中进行多轮测试，统计成功率以判定是否可以进入下一难度阶段。

    Args:
        model: 当前训练的 SB3 模型。
        env: 向量化评估环境。
        n_episodes: 评估的回合数，默认为 30。

    Returns:
        tuple: (平均物理奖励 mean_reward, 阶段成功率 success_rate)
    """
    successes = []  # 用于存储每个 episode 是否成功的列表

    def callback(local_vars, global_vars):
        for i, done in enumerate(local_vars['dones']):  # local_vars['dones'] 包含了内部的 infos, dones, rewards 等，是一个布尔数组 (因为是 VecEnv)
            if done:
                info = local_vars['infos'][i]
                if "is_success" in info:  # 从 info 中提取是否成功
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

    success_rate = np.mean(successes) if successes else 0.0

    return mean_reward, success_rate


def find_latest_checkpoint(find_dir: str) -> tuple[str | None, str | None, int]:
    """从指定目录中寻找最新的模型存档和环境归一化状态并匹配课程阶段，用于断点续训。

    Args:
        find_dir (str): 需要搜索存档的目录路径。

    Returns:
        tuple: 包含三个元素：
            - model_path (str | None): 最新的模型文件 (.zip) 路径，未找到则为 None。
            - stats_path (str | None): 最新的归一化状态文件 (.pkl) 路径，未找到则为 None。
            - stage_idx (int): 从文件名中提取的课程阶段索引，未找到或提取失败则默认返回 0。
    """
    model_files = glob.glob(os.path.join(find_dir, "*.zip"))
    stats_files = glob.glob(os.path.join(find_dir, "*.pkl"))

    goon_model_path = None
    goon_stats_path = None
    stage_idx = 0

    if model_files and stats_files:
        # 按文件修改时间排序，取最新的一个
        goon_model_path = max(model_files, key=os.path.getmtime)
        goon_stats_path = max(stats_files, key=os.path.getmtime)
        print(f"检测到存档，准备断点续训...")
        print(f"加载模型: {os.path.basename(goon_model_path)}")
        
        # 从文件名中提取 stage_idx 
        try:
            match = re.search(r"stage-(\d+)", os.path.basename(goon_stats_path))
            if match:
                stage_idx = int(match.group(1))
                print(f"已恢复至课程阶段: {stage_idx}")
        except Exception as e:
            print(f"警告⚠️：未能从文件名提取 stage_idx, 将使用默认值 0. 错误: {e}")
            
    return goon_model_path, goon_stats_path, stage_idx


def train():
    """主训练循环配置与执行器。
    
    包含架构逻辑：
    1. 路径配置 (实验结果的结构化保存路径`/outputs`、"断点续训"寻址)。
    2. 环境配置：
        `env`训练环境构建 (Monitor层、SubprocVecEnv 并行加速+向量化、VecNormalize 归一化观测空间)+断点续训。
        `eval_env`独立的评估环境 (Monitor层、DummyVecEnv 单一环境+向量化、VecNormalize 归一化观测空间)。
    3. 课程学习 (Curriculum Learning) 状态流转控制器。

    评估体系：
    EvalCallback：
    保存 best，防止过拟合/崩溃；
    强制智能体关闭探索噪声（即设置了 deterministic=True），每次都选当前认为最优的动作；只有 EvalCallback 生成的 eval/mean_reward 曲线，才是智能体当下真正的能力体现。如果没有它，你看着 TensorBoard 上抖动的训练曲线，根本不知道模型到底收敛了没有。

    两者各司其职：
    EvalCallback：是平时的**“期中测验”**。负责在漫长的学习过程中，定期摸底、记录真实成绩曲线，并帮你把考得最好的一次试卷复印件（Best Model）偷偷存下来防备万一。
    course_evaluate 函数：则是关卡末尾的**“期末大考”**。决定了你能不能升到下一个墙体高度（下一阶段课程）。
    """
    # ------------ 1. 路径配置 ------------ 
    cfg = TrainConfig()

    # save 路径（实验结果的结构化保存）
    output_dir = cfg.output_dir
    best_dir = os.path.join(output_dir, "best")
    latest_dir = os.path.join(output_dir, "latest")
    stages_dir = os.path.join(output_dir, "stages")
    tb_log_dir = os.path.join(output_dir, "tb_logs")
    for d in [best_dir, latest_dir, stages_dir, tb_log_dir]:
        os.makedirs(d, exist_ok=True)

    # "断点续训"寻址
    # 【choose：latest_dir 或 best_dir 或 stages_dir】; 默认 latest_dir
    find_dir = latest_dir
    # `find_latest_checkpoint`：获取最新 .zip 和 .pkl 路径并同步课程阶段
    goon_model_path, goon_stats_path, stage_idx = find_latest_checkpoint(find_dir)

    # ------------ 2. 环境配置 ------------ 
    """
    训练阶段：
        并行环境，使用 SubprocVecEnv
        Monitor层、SubprocVecEnv 并行加速+向量化、VecNormalize 归一化观测空间)+断点续训。
    评估阶段：
        不需要并行，使用 DummyVecEnv，更稳定且方便提取 info
        独立于 train 环境
        training=False 表示不更新统计值
        norm_reward=False 不归一化奖励
        callback_after_eval=sync_callback 同步归一化参数
    """
    def make_env(rank: int, seed: int = 0):
        """
        环境工厂函数：为每个进程创建一个独立的环境实例
        """
        def _init():
            env = PlanarBringBallEnv(model_path=cfg.xml_path)
            # 为每个环境设置不同的随机种子，非常重要！
            env.reset(seed=seed + rank)
            env = Monitor(env)
            return env
        return _init
    
    # 2.1 `env`训练环境
    # venv = DummyVecEnv([make_env])
    # 注意：在 MacOS/Linux 上建议显式指定 start_method
    n_envs = cfg.n_envs  
    venv = SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method='forkserver')

    # 断点续训
    if goon_model_path and os.path.exists(goon_model_path):
        env = VecNormalize.load(goon_stats_path, venv)
        model = PPO.load(goon_model_path, env=env)
    else:
        print(f"未检测到存档，开始全新训练...")
        env = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.)
        model = PPO("MlpPolicy", env,  # 不共享 Critic Network 和 Actor Network 参数 # "MlpPolicy" 默认会构建两套完全独立的多层感知机（MLP）
                    learning_rate=cfg.learning_rate,
                    n_steps=cfg.n_steps,   # 单个env采集 experience 数量
                    batch_size=cfg.batch_size,
                    verbose=1,
                    tensorboard_log=tb_log_dir)

    # 2.2 `eval_env`评估环境
    eval_venv = DummyVecEnv([make_env(rank=0)])
    eval_env = VecNormalize(eval_venv, training=False, norm_reward=False)
    eval_env.obs_rms = env.obs_rms

    # ------------ 3. 课程学习 ------------ 
    """
    存档: 
        最新数据（覆盖逻辑）—用于断点续训：位于 outputs/latest/
        各课程阶段的数据（全保存）：位于 outputs/stages/
    晋级标准：
        通过 course_evaluate 函数,在评估环境测试 30 组 episode, 若成功率大于 80%，则晋级
    """
    # 初始化 stage_idx (默认从0开始)
    # stage_idx = 0 # 在`find_latest_checkpoint`中实现，故注释掉
    total_stages= len(cfg.curriculum_stages)

    while stage_idx < total_stages:
        # 1. 获取课程阶段并更新环境难度
        stage = cfg.curriculum_stages[stage_idx]
        print(f"\n================ [阶段 {stage_idx}] 墙高: {stage['wall_height']} ================")

        env.env_method("set_wall_height", stage['wall_height'])
        eval_env.env_method("set_wall_height", stage['wall_height'])

        # 2. Callbacks
        # save: best/*.zip & *.pkl(覆盖逻辑，仅最后阶段保存，减小计算开销)
        # Callback1_1: SaveVecNormalizeCallback
        if_save_best = (stage_idx == total_stages - 1)
        best_model_dir = None
        save_vec_callback = None

        if if_save_best:
            print(f"检测到当前为最后阶段 (Stage {stage_idx})，将开启最优模型记录...")
            best_model_dir = best_dir
            b_total_steps = model.num_timesteps
            b_name = f"stage-{stage_idx}_step-{b_total_steps}"
            b_stats_filename = f"vec-normalize_{b_name}.pkl"

            for file in os.listdir(best_dir):
                file_path = os.path.join(best_dir, file)
                try:
                    if os.path.isfile(file_path): os.unlink(file_path)
                except Exception as e: print(f"清理失败: {e}")

            best_stats_path = os.path.join(best_dir, b_stats_filename)
            save_vec_callback = SaveVecNormalizeCallback(save_path=best_stats_path)

        # Callback1_main: EvalCallback
        eval_callback = EvalCallback(
            eval_env, 
            best_model_save_path=best_model_dir,   # save best model(.zip)；EvalCallback中此参数应该是 zip 的上级文件夹
            log_path=output_dir, 
            eval_freq=cfg.eval_freq_steps, # 注：eval_freq 监控的是单环境调用 env.step() 的次数！故无需 * n_envs
            deterministic=True, 
            render=False,
            callback_on_new_best=save_vec_callback, # 同步 save best 归一化参数(.pkl)
        )

        # Callback2_main: InfoLoggerCallback
        info_callback = InfoLoggerCallback()

        # 3. 训练
        model.learn(
            total_timesteps=cfg.total_timesteps, 
            reset_num_timesteps=False,   # 保证 Tensorboard 曲线的连续性，不会在进入新课程时从 0 开始
            tb_log_name=f"stage_{stage_idx}",  
            callback=[eval_callback, info_callback]
        )

        # 4. 阶段结课评估
        print("结课评估：正在同步`eval_env`的观测空间归一化参数...")
        eval_env.obs_rms = env.obs_rms  # 关建 Debug！

        mean_reward, success_rate = course_evaluate(model, eval_env, n_episodes=30)
        print(f"阶段 {stage_idx} 结课评估完成 | 成功率: {success_rate:.2%} | 平均奖励: {mean_reward:.2f}")

        # 保存文件名格式
        current_total_steps = model.num_timesteps
        timestamp = time.strftime("%H%M") 
        base_name = f"stage-{stage_idx}_step-{current_total_steps}_mean_reward-{mean_reward:.2f}_{timestamp}"
        model_filename = f"model_{base_name}.zip"
        stats_filename = f"vec-normalize_{base_name}.pkl"

        # save: latest/*.zip & *.pkl
        print(f"正在更新 latest 存档...")
        latest_model_path = os.path.join(latest_dir, model_filename)
        latest_stats_path = os.path.join(latest_dir, stats_filename)
        model.save(latest_model_path)
        env.save(latest_stats_path)

        # 晋级判断
        if success_rate >= 0.8:
            # save: stages/*.zip & *.pkl
            print(f"🎉 成功率 {success_rate:.2%} 达标，保存至 stages/ 并晋级！")
            stage_save_path = os.path.join(stages_dir, model_filename)
            stage_stats_path = os.path.join(stages_dir, stats_filename)
            model.save(stage_save_path)
            env.save(stage_stats_path)
            
            stage_idx += 1
        else:
            print(f"❌ 成功率未达标，将在本阶段继续训练。")
            # 可以在这里根据需要调整学习率，防止在同一关卡卡死
            # model.learning_rate = 1e-4 

if __name__ == "__main__":
    train()