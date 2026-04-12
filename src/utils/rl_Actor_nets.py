import torch
import torch.nn.functional as F
from torch.distributions import Normal

"""
    DDPG：输出可导的确定性动作
    SAC-continuous：输出不可导的高斯分布的均值和标准差+重参数化采样得到动作
"""


class PolicyNet(torch.nn.Module):
    """
    Stochastic Policy + Discrete Action输出头
    适用于：SAC-concrete；TRPO-concrete；PPO-concrete
    特征：softmax 输出头
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1) # 离散动作输出头特有的 softmax


class PolicyNet(torch.nn.Module):
    """
    Deterministic Policy + Continuous Action输出头
    适用于：DDPG
        DDPG 算法特征：基于 max Q 框架，需要动作梯度。
        动作梯度: 确定性策略天然提供
        PS：注意神经网络的输出的是固定的，因此需要在外部给动作添加噪声添加探索性
    """
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound # tanh：[-1,1]


class PolicyNetContinuous(torch.nn.Module):
    """
    Stochastic Policy + Continuous Action输出头
    适用于：SAC-continuous
        SAC 特征：基于 max Q 框架，需要动作梯度。
        动作梯度: 重参数化采样。
        PS：神经网络的输出是随机的，自带探索性
    PS：这种重参数化采样的技巧不适用于 PPO 这类 Likelihood Ratio梯度的算法
    """
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        """
        重参数化采样：
        基于rsample(): a = μ + σ ⋅ ϵ，其中 ϵ ∼ N(0,1)，不依赖于网络参数。
        由此 a 对 μ 和 σ 都是可导的，满足 SAC 需要的动作梯度；
        """
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))

        # 重参数化，保证梯度可以回传
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()重参数化采样
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample) # tanh(): 把动作压缩到[-1,1]

        # 修正Loss Function 中的熵惩罚项：log_prob，解决了梯度准不准确的问题（数值问题）
        # PS：loss Function中包含了 log_prob 的项，而 log_prob 的计算中包含了 tanh()会对分布的概率密度空间造成压缩
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound

        # 返回动作和对应的对数概率密度
        return action, log_prob
    

class PolicyNetContinuous(torch.nn.Module):
    """
    Stochastic Policy + Continuous Action输出头
    适用于：TRPO-continuous；PPO-continuous
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std  # 高斯分布的均值和标准差