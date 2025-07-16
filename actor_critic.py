import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    """PPO 的 Actor-Critic 網路"""
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Actor 網路
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )
        # Critic 網路
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        # 動作分佈的標準差 (可學習參數)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        # 評估狀態價值
        value = self.critic(state)
        # 取得動作分佈的平均值
        mu = self.actor(state)
        # 建立常態分佈
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value