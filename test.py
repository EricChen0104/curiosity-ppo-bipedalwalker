import torch
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym
import numpy as np
import time

ENV_NAME = 'BipedalWalker-v3'
MODEL_PATH = './model/ppo_bipedalwalker_icm.pth'
NUM_EPISODES = 10 
DEVICE = torch.device("cpu") 

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        value = self.critic(state)
        mu = self.actor(state)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value


def test():
    env = gym.make(ENV_NAME, render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = ActorCritic(state_dim, action_dim).to(DEVICE)

    try:
        agent.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"模型權重從 {MODEL_PATH} 載入成功。")
    except FileNotFoundError:
        print(f"錯誤：找不到模型檔案 {MODEL_PATH}。請先執行訓練程式。")
        return
    except Exception as e:
        print(f"載入模型時發生錯誤: {e}")
        return

    agent.eval()

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        done = False
        total_reward = 0
        episode_length = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                dist, _ = agent(state_tensor)
                action = dist.mean 

            action_np = action.squeeze().cpu().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            total_reward += reward
            episode_length += 1
            state = next_state
            
            time.sleep(0.01)

        print(f"回合: {episode + 1}, 總獎勵: {total_reward:.2f}, 步數: {episode_length}")

    env.close()

if __name__ == '__main__':
    test()