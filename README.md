# 强化学习 学习过程记录 & 辅修课程“数据驱动的智能控制”⼤作业   —**上海大学 郑群 23122932**  

### 🎬 Task3平面机械手抓球任务—调试过程训练效果演示  

<div align="center">
<img src="Task3_manipulator_bring_ball/show_results/Vedio1_test_环境测试.gif" width="80%">
<p><b>Vedio1_test_环境测试</b></p>
</div>

<br/>

<table style="width: 100%; border-collapse: collapse; border: none;">
  <tr style="border: none;">
    <td width="33.3%" align="center" style="border: none;">
      <img src="Task3_manipulator_bring_ball/show_results/Vedio2_debug_夹爪不闭合.gif" width="100%">
      <br><sub>Vedio2_debug_夹爪不闭合</sub>
    </td>
    <td width="33.3%" align="center" style="border: none;">
      <img src="Task3_manipulator_bring_ball/show_results/Vedio3_debug_夹爪提前闭合.gif" width="100%">
      <br><sub>Vedio3_debug_夹爪提前闭合</sub>
    </td>
    <td width="33.3%" align="center" style="border: none;">
      <img src="Task3_manipulator_bring_ball/show_results/Vedio4_debug_成功包裹但球在手内剧烈震颤.gif" width="100%">
      <br><sub>Vedio4_debug_成功包裹但球在手内剧烈震颤</sub>
    </td>
  </tr>
  <tr style="border: none;">
    <td width="33.3%" align="center" style="border: none;">
      <img src="Task3_manipulator_bring_ball/show_results/Vedio5_debug_成功抓住并短时间举起但会掉.gif" width="100%">
      <br><sub>Vedio5_debug_成功抓住并短时间举起但会掉</sub>
    </td>
    <td width="33.3%" align="center" style="border: none;">
      <img src="Task3_manipulator_bring_ball/show_results/Vedio6_debug_成功送至目标点但没有停下来.gif" width="100%">
      <br><sub>Vedio6_debug_成功送至目标点但没有停下来</sub>
    </td>
    <td width="33.3%" align="center" style="border: none;">
      </td>
  </tr>
</table>

<br/>

<div align="center">
<img src="Task3_manipulator_bring_ball/show_results/Vedio7_success.gif" width="80%">
<p><b>Vedio7_第一次成功（对应`--wall 0.00 --exp_name "v6.0_exp-01_PPO" --choose_model "latest" --match_id 69.09`）</b></p>
</div>

<br/>

<table style="width: 100%; border-collapse: collapse; border: none;">
  <tr style="border: none;">
    <td width="50%" align="center" style="border: none;">
      <img src="Task3_manipulator_bring_ball/show_results/Vedio8_wall_0.00.gif" width="100%">
      <br><sub>Vedio8_wall_0.00（对应`--wall 0.00 --exp_name "v6.1_exp-01_PPO" --choose_model "stages" --match_id stage-0`）</sub>
    </td>
    <td width="50%" align="center" style="border: none;">
      <img src="Task3_manipulator_bring_ball/show_results/Vedio9_wall_0.05.gif" width="100%">
      <br><sub>Vedio9_wall_0.05（对应`--wall 0.05 --exp_name "v6.1_exp-01_PPO" --choose_model "stages" --match_id stage-1`）</sub>
    </td>
  </tr>
  <tr style="border: none;">
    <td width="50%" align="center" style="border: none;">
      <img src="Task3_manipulator_bring_ball/show_results/Vedio10_wall_0.10.gif" width="100%">
      <br><sub>Vedio10_wall_0.10（对应`--wall 0.10 --exp_name "v6.1_exp-01_PPO" --choose_model "stages" --match_id stage-2`）</sub>
    </td>
    <td width="50%" align="center" style="border: none;">
      <img src="Task3_manipulator_bring_ball/show_results/Vedio11_wall_0.25.gif" width="100%">
      <br><sub>Vedio11_wall_0.25（对应`--wall 0.25 --exp_name "v6.1_exp-01_PPO" --choose_model "stages" --match_id stage-3`）</sub>
    </td>
  </tr>
</table>

---

## Details  

**Task3(Key): 基于给定 xml(略作修改)，从 0 实现“平面机械手跨越障碍抓球“**：  

```
rl_learn_zq_native/
├── Task3_manipulator_bring_ball/    # 【Task3】：从 0 实现“平面机械手跨越障碍抓球“
│   ├── xml/                               # 仿真建模
│   │   ├── manipulator_bring_ball.xml     # 核心模型 xml
│   │   └── test_xml.py                    # 核心模型 xml 导入测试
│   ├── outputs/exp_xxx                    # 实验原始结果
│   │   ├── evaluations.npz
│   │   ├── tb_logs/
│   │   ├── best/
│   │   ├── latest/
│   │   └── stages/
│   ├── show_results/                      # 成果展示
│   │   └── *.gif/
│   ├── env.py                             # 自定义：仿真环境封装接口
│   ├── config.py                          # 自定义：全局配置
│   ├── train.py                           # 自定义：训练脚本
│   └── show.py                            # 自定义：演示 & 视频录制脚本
```

- DeepMind Control Suite: Manipulator 是一个经典的连续控制强化学习（Reinforcement Learning, RL）基准测试环境。它基于 MuJoCo 物理引擎构建，旨在模拟和解决高维度、复杂的机械臂操作问题。  
- 刚刚学习完强化学习基础理论，为了感受“奖励函数的定义”“课程学习”等等强化学习领域的常用技巧，本实验基于 DeepMind Control Suite: Manipulator 提供的 xml 模型文件，并在此基础上加入障碍（可以变化高度的墙）；通过强化学习标准算法库 Stable Baselines3 以及物理引擎 MuJoCo，从 0 实现“平面-机械手：操控物体-跨障-送至目标位置”任务。  
- “平面-机械手：操控物体-跨障-送至目标位置”是一个二维平面（x-z）的抓取、避障、搬运任务。智能体（Agent）需要学习一套控制策略，驱动机械臂从随机的初始状态出发，跨越不同高度的障碍墙，准确抓取（或推动）球体，并将其运送至墙另一侧的目标位置。  

> 任务原型(略微修改): DeepMind Control Suite: Manipulator  
> 原始xml来源(略微修改): https://github.com/Motphys/MotrixLab/blob/main/motrix_envs/src/motrix_envs/basic/manipulator/manipulator_bring_ball.xml  

### Tensorboard (Task3)  
``` zsh
cd projects_mac/own/rl_learn_zq_native/
conda activate rl_learn
tensorboard --logdir=Task3_manipulator_bring_ball/outputs/
```

### 实验结果展示 & 录制 (Task3)  
> 通过 --help 指令获取：**“train 阶段使用的参数“**；**“详细的指令和推荐的范围“**  
> 如果不能调用 mjpython 可以尝试 python  

```zsh
mjpython Task3_manipulator_bring_ball/show.py --help
```

```zsh
# 最简化示例
# 注意--exp_name别错了！！！在 outputs 文件夹里
mjpython Task3_manipulator_bring_ball/show.py --exp_name "v6.1_exp-01_PPO" --choose_model "stages" --match_id stage-3
```

```zsh
# 完整指令示例（演示）
mjpython Task3_manipulator_bring_ball/show.py --wall 0.250 --ball 0.300 0.032 --target -0.250 0.400 --exp_name "v6.1_exp-01_PPO" --choose_model "stages" --match_id stage-3
```

```zsh
# 完整指令示例（录制）
mjpython Task3_manipulator_bring_ball/show.py --wall 0.250 --ball 0.300 0.032 --target -0.250 0.400 --exp_name "v6.1_exp-01_PPO" --choose_model "stages" --match_id stage-3 --mode video --fps 100
```

---

**Task1: 基于手写算法，跑通 CartPole，Pendulum，并观察结果。从而感受各个算法的性能并了解底层组件的定义**：  

```
rl_learn_zq_native/
├── Task1_myAlgo/       # 【Task1】：基于手写算法，跑通 CartPole，Pendulum，并观察结果。从而感受各个算法的性能并了解底层组件的定义。          
│   ├── CartPole.ipynb/           # 对比离散型动作输出算法；DQN，Actor-Critic，TPRO-concrete，PPO-concrete，SAC-concrete
│   └── Pendulum.ipynb/           # 对比连续型动作输出算法：TPRO-continuous，PPO-continuous，SAC-continuous  
├── src/                # 手写 RL 算法库
│   ├── agents/                   # PPO, SAC, DQN, TRPO 等
│   └── utils/                    # 工具函数 & 神经网络骨架(Actor/Critic)对比
```

- 任务概述：基于手写算法，跑通 CartPole，Pendulum，并观察结果。从而感受各个算法的性能并了解底层组件的定义。  
    - CartPole：对比离散型动作输出算法： DQN，Actor-Critic，TPRO-concrete，PPO-concrete，SAC-concrete  
    - Pendulum：对比连续型动作输出算法： TPRO-continuous，PPO-continuous，SAC-continuous  

> 手写算法核心逻辑参考：上海交通大学 张伟楠 《动手学强化学习》 & https://github.com/boyu-ai/Hands-on-RL    

---

**Task2【删除】**：  

```
rl_learn_zq_native/
├── Task2_sb3_sop/      # 【Task2】：尝试使用 Stable Baselines3 框架
```

- 核心目标: 熟悉 SB3  
- 任务概述：尝试使用 Stable Baselines3 框架, 解决稍微困难些的任务 PandaReach-v3  

---

**Task4【TODO】：基于Linkerhand O6 视觉动力学的研究**:  

```
rl_learn_zq_native/
├── Task4_mylinker/             # Task4: Linkerhand O6 视觉动力学 [TODO]
│   └── xml/meshes/             # 初步转化好了 xml，并通过 mujoco 可视化 [TODO: debug]
```

---

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

#### 其他可选项

> Mac: (可选)录制视屏  
```zsh
# (可选) Mac 录制视屏
brew install ffmpeg
```

```zsh
# mov 转 gif
for f in *.mov; do ffmpeg -i "$f" -vf "fps=18,scale=-1:600" -sws_flags lanczos -fps_mode cfr -an "${f%.mov}.gif"; done
```


> WSL/Windows: Install pytorch  
```bash
#  示例：WSL/Windows + RTX 5060Ti (sm_120)：CUDA 12.8 | torch 2.7.0
# 避免--index-url指令冲突。不在requirements中安装
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

> Task2：需额外安装 pybullet  
```bash
# pip 不好装 pybullet，用 conda
conda install -c conda-forge pybullet
```
