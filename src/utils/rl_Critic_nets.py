"""3种Critic网络"""
import torch
import torch.nn.functional as F


class QValueNet(torch.nn.Module):
    ''' 
    适用于: 基于 Qvalue的算法+离散动作输出；如 DQN、SAC-concrete
    本质：s to Q(s,a_i)的映射
    '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # s；输入层维度：（Batch size, state_dim）
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)  # Q(s,a_i)；输出层维度：（Batch size, action_dim）

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class QValueNetContinuous(torch.nn.Module):
    '''
    适用于: 基于 Qvalue的算法+连续动作输出；如 DDPG、SAC-continuous
    本质：s,a to Q(s,a)的映射
    '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim) # s + a
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)  # Q(s,a)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1) # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
    
class ValueNet(torch.nn.Module):
    '''
    适用于：基于 Policy Gradient；如 Actor-Critic,TRPO,PPO(concrete & continuous)
    本质：s to V(s)的映射
    Skill: Advantage Function A(s,a) = Q(s,a) - V(s) 
    '''
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim) # s
        self.fc2 = torch.nn.Linear(hidden_dim, 1) # V(s)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)