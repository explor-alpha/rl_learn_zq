"""
config.py: 
    对应任务: 平面机械手抓球——DeepMind Control Suite 中的 Manipulator 经典任务
    对应 MuJoCo 的模型实例: "manipulator_bring_ball.xml" 
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class TrainConfig:
    # --- 实验元信息 ---
    exp_name: str = "v6.0_exp-01_PPO"
    task_dir: str = os.path.dirname(os.path.abspath(__file__))
    xml_path: str = os.path.join(task_dir, "xml", "manipulator_bring_ball.xml")
    output_dir: str = os.path.join(task_dir, "outputs", exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # Simulation
    ctrl_dt: float = 0.01  # 根据 real 频率确定
    sim_dt: float = 0.002  # 控制物理引擎计算进度；先选 sim_substeps 并确保 ctrl_dt/sim_dt 整除 # 用于顶替xml中的频率
    sim_substeps: int = 5 # ctrl_dt/sim_dt 必须整除
    episode_max_time: float = 10.0 # 单个 episode 最长时间
    episode_max_steps: int = 1000   # 单个 episode 最大步数（env.py）# episode_max_time / ctrl_dt

    # --- DRL 相关参数 --- 
    # 启动 CPU 并行运算
    n_envs = 16  # WSL CPU；使用并行环境数量

    # 神经网络参数更新和总训练步数
    n_steps: int = 2048   # train.py, n_steps * n_envs 后进行一次梯度回传更新参数
    eval_freq_steps: int = (2048 * 16) * 5  # train.py， 更新 (n_steps * n_envs) * 5 次后调用 hook 评估训练效果
    total_timesteps: int = (2048 * 16) * 10  # 1000000  # train.py，learn，total_timesteps后评估一次课程训练效果，判断是否进入下一课程或终止

    # PPO 超参
    batch_size: int = 512   # 16 个环境并行。一次梯度更新拥有 n_steps * n_envs条数据。 512 依然属于中等偏小的尺寸，非常安全，不容易引起过拟合。
                            # 且较大的 batch size 抵消数据的特殊性（比如刚好这几次都摔倒了），让学习曲线更平滑。
    learning_rate: float = 3e-4
    gamma: float = 0.99

    # --- reward 相关参数 ---
    # 阈值
    pause_grasp2b_threshold: float = 0.200 
    touch_sensor_threshold: float = 0.01
    lift_height_threshold: float = 0.040  # 球被抬起的高度阈值，超过这个高度视为成功抬起
    success_dist_threshold: float = 0.005

    # 奖励权重
    # discount_post_grasp: float = 0.7  # 建议设置 0.5~0.9

    reach_weight: float = 1.0    # 1.0
    orient_weight: float = 1.5   # 1.5
    pause_weight: float = 0.5    # 0.5
    close_weight: float = 3.0    # 2.0
    lift_reward_weight: float = 1.0  # 1.0

    transport_weight: float = 6.0    # 6.0
    precision_weight: float = 1.0    # 1.0

    transport_progress_scale: float = 2.0    # 2.0
    hover_penalty_scale: float = 0.00    # 0.02

    # reward_success: float = 100.0  # 不能太小避免淹没；不能太大 Critic 网络梯度爆炸；尝试 50-200

    # --- 课程学习配置 ---
    curriculum_stages: List[Dict] = field(default_factory=lambda: [
        {"wall_height": 0.00},
        {"wall_height": 0.05},
        {"wall_height": 0.10},
        {"wall_height": 0.25},
    ])

