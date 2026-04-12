> 核心逻辑参考：上海交通大学-《动手学强化学习》https://github.com/boyu-ai/Hands-on-RL  

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

```
conda create -n mujoco python=3.10
```

```
conda activate mujoco
```

``` 
# Windows + RTX 5060Ti (sm_120)：CUDA 12.8 | torch 2.7.0
# 避免--index-url指令冲突。不在requirements中安装
# pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

```zsh
# Macbook M5
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

# 验证
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
```

```
pip install -r requirements.txt
```