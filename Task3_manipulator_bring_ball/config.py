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
    exp_name: str = "exp-01_PPO"
    task_dir: str = os.path.dirname(os.path.abspath(__file__))

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

    # --- 自动生成：路径参数---
    output_dir: str = ""
    latest_model_path: str = ""
    latest_stats_path: str = ""
    best_model_path: str = ""
    xml_path: str = ""

    def __post_init__(self):
        """在初始化后自动构建路径"""
        # 输出根目录
        self.output_dir = os.path.join(self.task_dir, "outputs", self.exp_name)
        
        # 具体模型路径
        self.latest_model_path = os.path.join(self.output_dir, "latest_model")
        self.latest_stats_path = os.path.join(self.output_dir, "latest_vec_normalize.pkl")
        self.best_model_path = os.path.join(self.output_dir, "best_model")
        
        # 环境 XML 路径
        self.xml_path = os.path.join(self.task_dir, "xml", "manipulator_bring_ball.xml")
        
        # 自动创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)