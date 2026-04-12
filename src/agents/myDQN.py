import os, sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

import numpy as np
import torch
import torch.nn.functional as F

    
class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络;
    对于动作离散的情况, 可以是 s to Q(s,a_i)的映射, 也可以是 s,a to Q(s,a) 的映射；
    对于连续动作情况, 一般是 s,a to Q(s,a) 的映射
    此处采用 s to Q(s,a_i)的映射
    '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim) # 输入层维度：（Batch size, state_dim） # 另一种情况：(Batch size, state_dim + action_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim) # 输出层维度：（Batch size, action_dim）# 另一种情况：(Batch size, 1)

    def forward(self, x): # input： state # 另一种情况： state, action
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x) # 输出层不使用激活函数，直接输出Q值 # output： Q(s,a_i) # 另一种情况： Q(s,a)
    
class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        # 常规 RL 超参
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略

        # Pytorch：常规 DL 超参
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate) # 使用Adam优化器
        self.device = device

        # DQN：Target Network
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device) # 目标网络
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录 update 函数调用次数，更新次数

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1) # 下一个状态的最大Q值，greedy
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标 # 此处done=1时，即 trajectory 的末尾，只有reward，没有下一个状态的Q值
        
        # Pytorch：Loss & Optimization
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数 # 本质是q_values 和 q_targets的误差回归
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        # Hard Update：Target Network
        if self.count % self.target_update == 0: # 每当 self.count 达到 self.target_update 的整数倍时，条件成立
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1
