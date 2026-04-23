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
    exp_name: str = "v2_exp-01_PPO_r1"
    task_dir: str = os.path.dirname(os.path.abspath(__file__))
    xml_path: str = os.path.join(task_dir, "xml", "manipulator_bring_ball.xml")
    output_dir: str = os.path.join(task_dir, "outputs", exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # --- PPO 超参 ---
    learning_rate: float = 3e-4
    batch_size: int = 64
    gamma: float = 0.99

    # --- 步长相关 参数 ---
    episode_max_steps: int = 1000  # env.py，单个 episode 最大步数, 之后 truncated
    n_steps: int = 2048   # train.py, 2048 步数后进行一次梯度回传更新参数
    eval_freq_steps: int = 2048*4  # train.py，n_steps*4 调用 hook 评估训练效果
    total_timesteps: int = 1000000  # train.py，learn，total_timesteps后评估一次课程训练效果，判断是否进入下一课程或终止

    # --- reward 相关参数 ---
    # 成功终止条件
    success_threshold=0.05
    # 奖励权重
    reward_weight_1 = 1.0  
    reward_weight_2 = 0.5
    reward_weight_3 = 2.0
    reward_weight_success = 10.0
    
    # --- 课程学习配置 ---
    curriculum_stages: List[Dict] = field(default_factory=lambda: [
        {"wall_height": 0.00},
        {"wall_height": 0.10},
        {"wall_height": 0.20},
        {"wall_height": 0.30},
    ])