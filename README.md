### 项目目标

为了实现这个目标，将其分解成一下几个 Tasks 逐步完成

**Task1**：  
- 问题的定义：通过自定义的算法（DQN，DDPG，SAC，PPO 等等），跑通 CartPole（离散动作输出），Pendulum（连续动作输出） 环境。  
- 目标: 了解各个算法的性能和底层组件的定义  
- 核心逻辑参考： 上海交通大学-《动手学强化学习》https://github.com/boyu-ai/Hands-on-RL    

**Task2**：  
- 问题的定义：调用标准库 SB3 ，解决稍微困难些的问题 PandaReach-v3
- 目标：  
1. 尝试调用 SB3 ，了解其功能   

**Task3【Doing】**：  
- 假设给定一个定义好的环境（DeepMind Control Suite 中的 Manipulator 经典任务：平面机械手抓球），已知 xml  
- 探索基于 mojoco+SB3 的 RL 任务训练流程：尝试自定义环境（observation，reward等等逻辑），自定义训练脚本（调用 SB3；课程学习；Callback-tensorboard等等逻辑）  
- 添加可变动高度的隔板，感受课程学习的能力  

**Task4【TODO】**:  
- 基于Linkerhand O6 视觉动力学的研究  

### 项目文件结构

```
rl_learn_zq_native/
├── src/                        # 手写强化学习算法库（主要逻辑参考“动手学强化学习”）
│   ├── agents/                 # PPO, SAC, DQN, TRPO 等算法的 Native 实现
│   └── utils/                  # 神经网络(Actor/Critic)构筑与 RL 工具函数
├── Task1_myAlgo/               # Task1：在经典环境(CartPole/Pendulum)验证手写算法
├── Task2_sb3_sop/              # Task2：尝试使用 Stable Baselines3 框架
│   ├── outputs/                
│   └── PandaReach.ipynb        # 基于 Panda 机械臂的达标任务实验
├── Task3_manipulator_bring_ball/ # Task3: 自定义 Mujoco 机械臂抓球任务 [Doing]
│   ├── xml/                    # 仿真场景建模 xml（copy from MotrixLab）
│   ├── config.py               # 自定义config
│   ├── env.py                  # 自定义仿真环境封装接口
│   └── train.py                # 自定义的训练脚本
├── Task4_mylinker/             # Task4: Linkerhand O6 视觉动力学 [TODO]
│   └── xml/meshes/             # 初步转化好了 xml，并通过 mujoco 可视化 [debug]
├── requirements.txt            
└── README.md      
```

参照 Google Python Style Guide（⾕歌注释规范） 编写注释

### 环境配置
#### Macbook M5环境配置（**Native**）
```zsh
conda create -n rl_learn python=3.10 -y
conda activate rl_learn

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
pip install -r requirements.txt

# setuptools默认版本可能和 tensorboard 不适配，手动降版本
pip install "setuptools<70"
```

```zsh
# 验证 mps
export KMP_DUPLICATE_LIB_OK=TRUE
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
``` 

#### 其他
```zsh
# (可选) Mac 录制视屏
brew install ffmpeg
```

```bash
# WSL/Windows + RTX 5060Ti (sm_120)：CUDA 12.8 | torch 2.7.0
# 避免--index-url指令冲突。不在requirements中安装
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

> PS:Task2 需额外安装 pybullet
```bash
# pip 不好装 pybullet，用 conda
conda install -c conda-forge pybullet
```

### Tensorboard
``` zsh
# 示例
cd projects_mac/own/rl_learn_zq_native/
conda activate rl_learn
tensorboard --logdir=Task3_manipulator_bring_ball/outputs/
```




