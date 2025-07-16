import torch
import torch.nn as nn

class ICM(nn.Module):
    """內在好奇心模組 (Intrinsic Curiosity Module)"""
    def __init__(self, state_dim, action_dim, feature_dim=256):
        super(ICM, self).__init__()
        # 狀態編碼器 (將狀態轉為特徵)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        # 反向模型 (根據 s, s' 預測 a)
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        # 前向模型 (根據 s, a 預測 s')
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, state, next_state, action):
        # 取得狀態特徵
        state_feature = self.encoder(state)
        next_state_feature = self.encoder(next_state)

        # 預測動作 (反向模型)
        pred_action = self.inverse_model(torch.cat((state_feature, next_state_feature), dim=1))

        # 預測下一個狀態的特徵 (前向模型)
        pred_next_state_feature = self.forward_model(torch.cat((state_feature, action), dim=1))

        return state_feature, pred_next_state_feature, next_state_feature, pred_action