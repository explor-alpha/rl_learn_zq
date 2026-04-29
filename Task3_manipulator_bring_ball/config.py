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
    exp_name: str = "v4_exp-10_PPO_r4"
    task_dir: str = os.path.dirname(os.path.abspath(__file__))
    xml_path: str = os.path.join(task_dir, "xml", "manipulator_bring_ball.xml")
    output_dir: str = os.path.join(task_dir, "outputs", exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # --- 启动 CPU 并行运算 ---
    n_envs = 16  # WSL CPU；使用并行环境数量

    # --- 步长相关 参数 ---
    episode_max_steps: int = 1000  # env.py，单个 episode 最大步数, 之后 truncated # 最大10s/xml中定义的 0.01 Hz=> 1000 步
    n_steps: int = 2048   # train.py, n_steps * n_envs 后进行一次梯度回传更新参数
    eval_freq_steps: int = (2048 * 16) * 5  # train.py， 更新 (n_steps * n_envs) * 5 次后调用 hook 评估训练效果
    total_timesteps: int = (2048 * 16) * 5  # 1000000  # train.py，learn，total_timesteps后评估一次课程训练效果，判断是否进入下一课程或终止

    # --- PPO 超参 ---
    batch_size: int = 512   # 16 个环境并行。一次梯度更新拥有 n_steps * n_envs条数据。 512 依然属于中等偏小的尺寸，非常安全，不容易引起过拟合。
                            # 且较大的 batch size 抵消数据的特殊性（比如刚好这几次都摔倒了），让学习曲线更平滑。
    learning_rate: float = 3e-4
    gamma: float = 0.99

    # --- reward 相关参数 ---
    # 成功终止条件
    success_lift_threshold_height: float = 0.04  # 球被抬起的高度阈值，超过这个高度视为成功抬起
    success_threshold: float = 0.05
    # 奖励权重
    reach_weight: float = 1.0
    orient_weight: float = 1.5   # 1.5
    pause_weight: float = 0.2    # 0.2
    close_weight: float = 2.0    # 2.0
    lift_reward_weight: float = 0.0
    transport_weight: float = 6.0    #6.0
    precision_weight: float = 1.0    #1.0
    reward_success: float = 10.0

    # --- 课程学习配置 ---
    curriculum_stages: List[Dict] = field(default_factory=lambda: [
        {"wall_height": 0.00},
        {"wall_height": 0.10},
        {"wall_height": 0.20},
        {"wall_height": 0.30},
    ])

