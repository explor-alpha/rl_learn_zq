"""强化学习训练辅助工具库.
核心逻辑参考：上海交通大学-《动手学强化学习》
https://github.com/boyu-ai/Hands-on-RL

本模块主要包含：
1. 辅助组件：ReplayBuffer，compute_advantage
2. train 逻辑：train_on_policy_agent, train_off_policy_agent
3. 绘制 Return 曲线：moving_average, plot_returns, plot_returns_combined
4. 可视化演示：show_live_performance

遵循 Google Python Style Guide。
"""

import collections
import random
from typing import Any, Dict, List, Tuple
import time
import pygame

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


class ReplayBuffer:
    """经验回放池，用于存储和采样离线策略算法的转场数据。

    1. Replay buffer的数据不会随 episode 的结束而清空，整个训练过程中不断累积数据
    2. 主要用于 DQN、SAC 和 DDPG 等离线策略算法。

    Attributes:
        buffer: 存储转场数据的双端队列。
    """

    def __init__(self, capacity: int):
        """初始化经验回放池。

        队列,先进先出

        Args:
            capacity: 缓冲池的最大容量。
        """
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: Any, reward: float, 
            next_state: np.ndarray, done: bool):
        """向缓冲池添加一条转场数据。

        Args:
            state: 当前状态。
            action: 采取的动作。
            reward: 获得的奖励。
            next_state: 下一个状态。
            done: 终止标志。
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """从缓冲池中随机采样一个批次的数据。

        Args:
            batch_size: 采样的批次大小。

        Returns:
            包含 (states, actions, rewards, next_states, dones) 的元组。
        """
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return (np.array(state), action, reward, np.array(next_state), done)

    def size(self) -> int:
        """返回当前缓冲池中的数据量。"""
        return len(self.buffer)


def train_on_policy_agent(
        env: Any, 
        agent: Any, 
        num_episodes: int, 
        seed: int=None
        ) -> List[float]:
    """在线策略(On-policy)算法训练循环。

    适用于 PPO, Actor-Critic 等算法。每一轮 Episode 结束后立即进行策略更新。

    Args:
        env: 遵循 Gymnasium 接口的环境。
        agent: 智能体实例，需具备 take_action(state) 和 update(transition_dict) 方法。
        num_episodes: 总训练回合数。
        seed: 随机种子。

    Returns:
        每个回合的累积奖励列表。
    """
    return_list = []
    # 分10个阶段展示进度
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    'states': [], 'actions': [], 'next_states': [], 
                    'rewards': [], 'dones': []
                }
                state, _ = env.reset(seed=seed)
                done = False
                
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    # 记录轨迹
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    
                    state = next_state
                    episode_return += reward
                
                return_list.append(episode_return)
                # 训练更新
                agent.update(transition_dict)
                
                # 进度条展示最近10场平均回报
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': f'{int(num_episodes/10 * i + i_episode + 1)}',
                        'return': f'{np.mean(return_list[-10:]):.3f}'
                    })
                pbar.update(1)
    return return_list


def train_off_policy_agent(
    env: Any, 
    agent: Any, 
    num_episodes: int, 
    replay_buffer: ReplayBuffer, 
    minimal_size: int, 
    batch_size: int,
    seed: int=None
) -> List[float]:
    """离线策略(Off-policy)算法训练循环。

    适用于 DQN, SAC, DDPG 等算法。数据存入 Buffer，当数据量足够时随机采样更新。

    Args:
        env: 遵循 Gymnasium 接口的环境。
        agent: 智能体实例。
        num_episodes: 总训练回合数。
        replay_buffer: 经验回放池实例。
        minimal_size: 开始训练前 Buffer 需达到的最小数据量。
        batch_size: 每次更新采样的样本数。
        seed: 随机种子。

    Returns:
        每个回合的累积奖励列表。
    """
    return_list = []
    # 将整个训练过程分为 10 个大阶段（Iteration）
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
            # 每个 iteration 玩 50 个 episode/trajectory
            for i_episode in range(int(num_episodes / 10)):
                # episode_return: 记录每条 trajectory rewards 的和，不参与算法更新，只是用来评估训练效果
                episode_return = 0
                state, _ = env.reset(seed=seed)
                done = False
                
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    # 存储经验
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    
                    # Replay buffer：当buffer数据的数量超过一定值后,才进行神经网络训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s, 
                            'actions': b_a, 
                            'next_states': b_ns,
                            'rewards': b_r, 
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                
                return_list.append(episode_return)
                
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': f'{int(num_episodes/10 * i + i_episode + 1)}',
                        'return': f'{np.mean(return_list[-10:]):.3f}'
                    })
                pbar.update(1)
    return return_list


def compute_advantage(gamma: float, lmbda: float, td_delta: torch.Tensor) -> torch.Tensor:
    """计算广义优势估计 (GAE)。

    使用 GAE 算法计算每个时间步的优势值。

    Args:
        gamma: 折扣因子。
        lmbda: GAE 平滑参数。
        td_delta: 由评论家网络(Critic)计算的时序差分误差。

    Returns:
        包含优势值的张量。
    """
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    # 从后往前逆序计算 GAE
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


def moving_average(a: List[float], window_size: int) -> np.ndarray:
    """计算数列的移动平均。

    该函数使用累加和方法高效计算移动平均值，并对边缘进行特殊处理。

    Args:
        a: 原始数值列表或数组。
        window_size: 移动平均的窗口大小。

    Returns:
        与输入长度相同的移动平均结果数组。
    """
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def plot_returns(return_list: List[float], 
                 env_name: str, 
                 algo_name: str, 
                 window_size: int = 9):
    """绘制训练回报曲线及移动平均曲线。

    该函数会生成两张图：一张是原始的回报曲线，另一张是经过平滑处理后的回报曲线。

    Args:
        return_list: 每个回合的累积奖励列表。
        env_name: 环境名称（用于标题显示）。
        algo_name: 算法名称。
        window_size: 移动平均的窗口大小，默认为 10。
    """
    episodes_list = list(range(len(return_list)))
    
    # 创建画布，包含两个子图（1行2列）
    plt.figure(figsize=(12, 5))

    # 子图1：原始回报
    plt.subplot(1, 2, 1)
    plt.plot(episodes_list, return_list, alpha=0.3, color='blue', label='Raw Return')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'{algo_name} on {env_name} (Raw)')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 子图2：移动平均回报
    plt.subplot(1, 2, 2)
    mv_return = moving_average(return_list, window_size)
    plt.plot(episodes_list, mv_return, color='red', label='Moving Average')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'{algo_name} on {env_name} (Smoothed)')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()  # 自动调整布局避免重叠
    plt.show()


def plot_returns_combined(return_list: List[float], 
                          env_name: str, 
                          algo_name: str, 
                          window_size: int = 9):
    """在同一张图内绘制原始曲线和移动平均曲线（对比更直观）。

    Args:
        return_list: 每个回合的累积奖励列表。
        env_name: 环境名称。
        algo_name: 算法名称。
        window_size: 移动平均窗口大小。
    """
    episodes_list = list(range(len(return_list)))
    mv_return = moving_average(return_list, window_size)

    plt.figure(figsize=(8, 6))
    plt.plot(episodes_list, return_list, alpha=0.2, color='gray', label='Raw Return')
    plt.plot(episodes_list, mv_return, color='red', linewidth=2, label='Moving Avg')
    
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'{algo_name} performance on {env_name}')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    

def show_live_performance(env_name: str, agent: Any, seed: int = None, episodes: int = 3):
    """可视化演示函数。
    
    Args:
        env_name: 环境名称。
        agent: 智能体实例。
        seed: 初始种子。
        episodes: 演示的回合数。
    """
    test_env = gym.make(env_name, render_mode="human")
    
    try:
        for i in range(episodes):
            # 为了让每一局都有点变化，给每一局不同的种子
            current_seed = seed + i if seed is not None else None
            state, _ = test_env.reset(seed=current_seed)
            
            done = False
            total_reward = 0
            
            while not done:
                # 处理 pygame 事件（防止窗口“未响应”）
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        test_env.close()
                        return

                action = agent.take_action(state)
                state, reward, terminated, truncated, _ = test_env.step(action)
                done = terminated or truncated
                total_reward += reward
                
                time.sleep(0.02)  # 控制播放速度
            
            print(f"回合 {i+1} 结束，得分: {total_reward}")
            time.sleep(0.5)
            
    finally:
        # 无论是否报错，最后都强制执行关闭操作
        test_env.close()
        # 核心修复：强制销毁 pygame 窗口
        pygame.display.quit()
        pygame.quit()
        print("可视化窗口已关闭")

