import os
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class TrainConfig:
    # --- 实验元信息 ---
    exp_name: str = "exp-00_PPO_debug_test011"
    # 获取当前文件所在目录作为 task_dir
    task_dir: str = os.path.dirname(os.path.abspath(__file__))
    
    # --- 基础 PPO 参数 ---
    total_timesteps: int = 512000  # 500000
    learning_rate: float = 3e-4
    n_steps: int = 2048   # 2048
    batch_size: int = 64
    gamma: float = 0.99
    
    # --- 课程学习配置 ---
    curriculum_stages: List[Dict] = field(default_factory=lambda: [
        {"wall_height": 0.00, "threshold": -50},
        {"wall_height": 0.15, "threshold": -40},
        {"wall_height": 0.30, "threshold": -30},
        {"wall_height": 0.45, "threshold": -20},
    ])

    # --- 自动生成的路径 (在 post_init 中处理) ---
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