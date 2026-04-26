# 强化学习 学习过程记录 & 辅修课程“数据驱动的智能控制”⼤作业   —**上海大学 郑群 23122932**  

> Task3 成果展示：  
> 基于给定 xml(略作修改)，从 0 实现：“平面机械手跨越障碍抓球“：

### 🎬 平面机械手抓球任务演示 (Task3)

> v2.1 训练效果对比演示

<p>从左到右依次为：延迟抓取、抓空球情况、成功抓取但未送达目标</p>

<!-- 第一张大图：独占一行 -->
<div align="center">
  <img src="Task3_manipulator_bring_ball/show_results/env_init.gif" width="80%" alt="环境初始化展示">
  <p><b>图 1：环境随机初始化（球与目标位置）</b></p>
</div>

<br/>

<!-- 后三张小图：并排一行 -->
<table style="width: 100%; border-collapse: collapse; border: none;">
  <tr style="border: none;">
    <td width="33.3%" align="center" style="border: none;">
      <img src="Task3_manipulator_bring_ball/show_results/v2.1_delay.gif" width="100%">
      <br><sub>延迟抓取</sub>
    </td>
    <td width="33.3%" align="center" style="border: none;">
      <img src="Task3_manipulator_bring_ball/show_results/v2.1_miss.gif" width="100%">
      <br><sub>抓空球情况</sub>
    </td>
    <td width="33.3%" align="center" style="border: none;">
      <img src="Task3_manipulator_bring_ball/show_results/v2.1_success.gif" width="100%">
      <br><sub>成功抓取</sub>
    </td>
  </tr>
</table>

---

## Details

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
│   │   └── *.mov/
│   ├── env.py                             # 自定义：仿真环境封装接口
│   ├── config.py                          # 自定义：全局配置
│   ├── train.py                           # 自定义：训练脚本
│   └── show.py                            # 自定义：演示 & 视频录制脚本
```

- 任务概述：从基于给定 xml(略作修改)，从 0 实现“平面机械手跨越障碍抓球“
    - 探索基于 mojoco+SB3 的 RL 任务训练流程：尝试自定义环境（observation，reward等等逻辑），自定义训练脚本（调用 SB3；课程学习；Callback-tensorboard等等逻辑）  
    - 添加可变动高度的隔板，感受课程学习的能力  
    - [TODO]

> 任务原型: DeepMind Control Suite: Manipulator
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
mjpython Task3_manipulator_bring_ball/show.py --mode human --wall 0.00 --exp_name "v2.1_exp-05_PPO_r5e"
```

```zsh
# 完整指令示例（演示）
mjpython Task3_manipulator_bring_ball/show.py --mode human --wall 0.00 --ball 0.30 0.03 --target -0.25 0.4 --exp_name "v2.1_exp-05_PPO_r5e" --choose_model "latest" --match_id 2002944

# 完整指令示例（录制）
mjpython Task3_manipulator_bring_ball/show.py --mode video --wall 0.30 --ball 0.30 0.03 --target -0.25 0.4 --steps 1000 --exp_name "v2.1_exp-05_PPO_r5e" --choose_model "latest" --match_id 2002944 --fps 90  
```

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
