### 项目目标

本项目最终要解决的问题是控制灵巧手抓取物体：
- observe：2 个相机
- action：6 个关节的目标位置，力矩，速度
- Assumption：加速度很慢，认为匀速
- Target：抓取乒乓球

为了实现这个目标，将其分解成一下几个 Tasks 逐步完成

Task1：  
- 问题的定义：通过自定义的算法（DQN，DDPG，SAC，PPO 等等），跑通 CartPole（离散动作输出），Pendulum（连续动作输出） 环境。  
- 目标: 了解各个算法的性能和底层组件的定义  
- 核心逻辑参考： 上海交通大学-《动手学强化学习》https://github.com/boyu-ai/Hands-on-RL    
Task2：  
- 问题的定义：调用标准库 SB3 ，解决稍微困难些的问题 PandaReach-v3
- 目标：  
1. 尝试调用 SB3 ，了解其功能   
2. 尝试调用 SB3 处理一个未知环境（例如 PandaReach-v3），并了解 SOP 流程 。如何探索环境？如何调用 SB3 选择算法？如何逐步调试到最终落实？等等  


### 项目文件结构

```数据驱动的智能控制 - 大作业
23122932  
 ┣ notebooks  
 ┃ ┣ CartPole.ipynb  # 执行脚本（CartPole环境）
 ┃ ┗ Pendulum.ipynb  # 执行脚本（Pendulum环境）
 ┣ src  
 ┃ ┣ agents  
 ┃ ┃ ┣ myActorCritic.py  
 ┃ ┃ ┣ myDQN.py  
 ┃ ┃ ┣ myPPO.py  
 ┃ ┃ ┣ myPPOcontinuous.py  
 ┃ ┃ ┣ mySAC.py  
 ┃ ┃ ┣ mySACcontinuous.py  
 ┃ ┃ ┣ myTRPO.py  
 ┃ ┃ ┗ myTRPOcontinuous.py  
 ┃ ┣ utils  
 ┃ ┃ ┣ rl_Actor_nets.py  
 ┃ ┃ ┣ rl_Critic_nets.py  
 ┃ ┃ ┗ rl_utils.py  
 ┣ README.md  
 ┗ requirements.txt
```

参照 Google Python Style Guide（⾕歌注释规范） 编写注释

### 环境配置
#### Macbook M5环境配置
```zsh
conda create -n mujoco python=3.10
conda activate mujoco
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
conda install -c conda-forge pybullet
pip install -r requirements.txt
pip install "setuptools<70"
```

#### Windows + RTX 5060Ti (sm_120)环境配置
> Windows + RTX 5060Ti (sm_120)：选择CUDA 12.8 | torch 2.7.0
```
conda create -n mujoco python=3.10
conda activate mujoco

# Windows + RTX 5060Ti (sm_120)：CUDA 12.8 | torch 2.7.0
# 避免--index-url指令冲突。不在requirements中安装
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

conda install -c conda-forge pybullet
pip install -r requirements.txt
```

#### Tensorboard
```
tensorboard --logdir=Reacher/logs/
```

#### 其他工具：

```Mac 录制视屏
brew install ffmpeg
```

