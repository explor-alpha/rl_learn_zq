"""config.py: """
import os
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class TrainConfig:
    # --- 实验元信息 ---
    exp_name: str = "v6.1_exp-03_PPO"
    task_dir: str = os.path.dirname(os.path.abspath(__file__))
    xml_path: str = os.path.join(task_dir, "xml", "manipulator_bring_ball.xml")
    output_dir: str = os.path.join(task_dir, "outputs", exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # --- Simulation, 物理仿真控制（env.py） ---
    ctrl_dt: float = 0.01  # 控制器时间步 (依据真实控制频率设定)
    sim_dt: float = 0.002  # 物理引擎时间步 (覆盖 XML 设置)
    sim_substeps: int = 5  # 子步数，需确保 ctrl_dt / sim_dt 整除
    episode_max_time: float = 10.0  # 单个 Episode 的最大物理时间 (秒)
    episode_max_steps: int = 1000   # 单个 episode 最大步数; = episode_max_time / ctrl_dt

    # --- DRL 系统参数 --- 
    # 每次收集完 n_steps * n_envs 条 experience 后进行梯度回传
    n_envs = 16  # CPU 并行环境数量
    n_steps: int = 2048   # 单个环境采集 experience 数量
    # 评估频率：每 5 个完整 rollout 进行一次模型评估
    eval_freq_steps: int = (2048) * 5  # 注：eval_freq 监控的是单环境调用 env.step() 的次数！故无需 * n_envs
    # 课程学习：total_timesteps后评估一次课程训练效果，判断是否进入下一课程或终止
    total_timesteps: int = (2048 * 16) * 15  # train.py，learn，

    # PPO 超参
    batch_size: int = 512   # 太大过拟合；太小无法抵消数据的特殊性，梯度不稳定；512：(2048 * 16) 适中
    learning_rate: float = 3e-4  # Actor-Critic 网络的初始学习率
    gamma: float = 0.99  # 折扣因子 (Discount Factor)

    # --- 奖励函数配置 (Reward Shaping) ---
    # 状态阈值
    pause_grasp2b_threshold: float = 0.200  
    touch_sensor_threshold: float = 0.01
    lift_height_threshold: float = 0.040  
    pause_b2t_threshold: float = 0.200     
    success_dist_threshold: float = 0.010

    # Phase 1: 抓取过程奖励权重
    reach_weight: float = 1.0        # 接近物体
    orient_weight: float = 1.5       # 姿态对齐
    pause_weight: float = 0.5        # 抓取前稳定停顿
    close_weight: float = 3.0        # 闭合抓取
    lift_reward_weight: float = 1.0  # 抬升物体

    # Phase 2: 运输与姿态稳定奖励权重
    transport_weight: float = 6.0    # 向目标点运输
    pause2_weight: float = 1.0       # 到达目标后的稳定停顿
    precision_weight: float = 1.0    # 目标点精准度

    # 势能与惩罚因子
    transport_progress_scale: float = 2.0  # 基于位移势能的密集奖励缩放
    hover_penalty_scale: float = 0.00      # 悬停而不抓取的惩罚因子

    # discount_post_grasp: float = 0.7  # 建议设置 0.5~0.9
    # reward_success: float = 50.0  # 不能太小避免淹没；不能太大 Critic 网络梯度爆炸；尝试 50-200

    # --- 课程学习 (Curriculum Learning) ---
    # 分阶段增加墙体高度，引导策略网络从简单任务逐步适应复杂越障任务。
    curriculum_stages: List[Dict] = field(default_factory=lambda: [
        {"wall_height": 0.00},
        {"wall_height": 0.05},
        {"wall_height": 0.10},
        {"wall_height": 0.25},
    ])

