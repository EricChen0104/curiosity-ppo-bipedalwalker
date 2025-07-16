import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from icm import ICM
from actor_critic import ActorCritic

DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")
ENV_NAME = 'BipedalWalker-v3'

GAMMA = 0.99                
GAE_LAMBDA = 0.95           
PPO_EPSILON = 0.2           
CRITIC_DISCOUNT = 0.5       
ENTROPY_BETA = 0.001        
PPO_EPOCHS = 10             
PPO_MINIBATCH_SIZE = 64     
PPO_LEARNING_RATE = 3e-4    

ICM_LEARNING_RATE = 1e-4    
INTRINSIC_REWARD_ETA = 0.02 
ICM_FORWARD_LOSS_BETA = 0.2 
ICM_INVERSE_LOSS_BETA = 0.8 

ROLLOUT_STEPS = 2048        
MAX_TRAINING_STEPS = 500_000 
PRINT_INTERVAL = ROLLOUT_STEPS * 10 

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards_e = [] 
        self.rewards_i = [] 
        self.values = []
        self.next_states = []
        self.dones = []

    def add(self, state, action, log_prob, reward_e, reward_i, value, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards_e.append(reward_e)
        self.rewards_i.append(reward_i)
        self.values.append(value)
        self.next_states.append(next_state)
        self.dones.append(done)

    def get_tensors(self):
        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(np.array(self.actions), dtype=torch.float32).to(DEVICE)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(DEVICE)
        rewards_e = torch.tensor(self.rewards_e, dtype=torch.float32).to(DEVICE)
        rewards_i = torch.tensor(self.rewards_i, dtype=torch.float32).to(DEVICE)
        values = torch.tensor(self.values, dtype=torch.float32).to(DEVICE)
        next_states = torch.tensor(np.array(self.next_states), dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(DEVICE)
        return states, actions, log_probs, rewards_e, rewards_i, values, next_states, dones

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards_e[:]
        del self.rewards_i[:]
        del self.values[:]
        del self.next_states[:]
        del self.dones[:]

def train():
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high[0]

    ppo_agent = ActorCritic(state_dim, action_dim).to(DEVICE)
    icm_module = ICM(state_dim, action_dim).to(DEVICE)
    
    ppo_optimizer = optim.Adam(ppo_agent.parameters(), lr=PPO_LEARNING_RATE)
    icm_optimizer = optim.Adam(icm_module.parameters(), lr=ICM_LEARNING_RATE)

    buffer = RolloutBuffer()

    state, _ = env.reset()
    total_steps = 0
    episode_num = 0

    all_rewards = []
    avg_rewards = []
    recent_scores = deque(maxlen=100)
    
    # 2. 開始訓練迴圈
    while total_steps < MAX_TRAINING_STEPS:
        for _ in range(ROLLOUT_STEPS):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                dist, value = ppo_agent(state_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(axis=-1)
                
            action_np = action.squeeze().cpu().numpy()
            next_state, reward_e, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(DEVICE)
                _, pred_next_state_feature, next_state_feature, _ = icm_module(state_tensor, next_state_tensor, action)
                reward_i = INTRINSIC_REWARD_ETA * 0.5 * ((pred_next_state_feature - next_state_feature)**2).sum(dim=1)

            buffer.add(state, action_np, log_prob.item(), reward_e, reward_i.item(), value.item(), next_state, done)
            
            state = next_state
            total_steps += 1

            if done:
                state, _ = env.reset()
                episode_num += 1

        states_t, actions_t, old_log_probs_t, rewards_e_t, rewards_i_t, values_t, next_states_t, dones_t = buffer.get_tensors()
        
        rewards_t = rewards_e_t + rewards_i_t
        
        advantages = torch.zeros_like(rewards_t)
        last_gae_lam = 0
        
        with torch.no_grad():
            _, last_value = ppo_agent(torch.FloatTensor(state).unsqueeze(0).to(DEVICE))
            last_value = last_value.squeeze()

        for t in reversed(range(ROLLOUT_STEPS)):
            if t == ROLLOUT_STEPS - 1:
                next_non_terminal = 1.0 - done 
                next_values = last_value
            else:
                next_non_terminal = 1.0 - dones_t[t+1]
                next_values = values_t[t+1]
            
            delta = rewards_t[t] + GAMMA * next_values * next_non_terminal - values_t[t]
            advantages[t] = last_gae_lam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lam
        
        returns = advantages + values_t 
        
        for _ in range(PPO_EPOCHS):
            sampler = torch.randperm(ROLLOUT_STEPS)
            for start in range(0, ROLLOUT_STEPS, PPO_MINIBATCH_SIZE):
                end = start + PPO_MINIBATCH_SIZE
                mb_indices = sampler[start:end]

                mb_states = states_t[mb_indices]
                mb_actions = actions_t[mb_indices]
                mb_old_log_probs = old_log_probs_t[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_next_states = next_states_t[mb_indices]

                dist, values = ppo_agent(mb_states)
                
                critic_loss = (mb_returns - values.squeeze()).pow(2).mean()

                new_log_probs = dist.log_prob(mb_actions).sum(axis=-1)
                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - PPO_EPSILON, 1.0 + PPO_EPSILON) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                entropy_loss = -dist.entropy().mean()

                ppo_loss = actor_loss + CRITIC_DISCOUNT * critic_loss + ENTROPY_BETA * entropy_loss
                
                ppo_optimizer.zero_grad()
                ppo_loss.backward()
                torch.nn.utils.clip_grad_norm_(ppo_agent.parameters(), 0.5)
                ppo_optimizer.step()

                _, pred_next_state_feature, next_state_feature, pred_action = icm_module(mb_states, mb_next_states, mb_actions)
                
                forward_loss = 0.5 * ((pred_next_state_feature - next_state_feature.detach())**2).mean()
                
                inverse_loss = nn.MSELoss()(pred_action, mb_actions.detach())

                icm_loss = (ICM_FORWARD_LOSS_BETA * forward_loss + ICM_INVERSE_LOSS_BETA * inverse_loss)
                
                icm_optimizer.zero_grad()
                icm_loss.backward()
                torch.nn.utils.clip_grad_norm_(icm_module.parameters(), 0.5)
                icm_optimizer.step()

        buffer.clear()
        
        eval_env = gym.make(ENV_NAME)
        eval_reward = 0
        eval_state, _ = eval_env.reset()
        eval_done = False
        while not eval_done:
            with torch.no_grad():
                s_tensor = torch.FloatTensor(eval_state).unsqueeze(0).to(DEVICE)
                dist, _ = ppo_agent(s_tensor)
                action = dist.mean # 在評估時使用確定性動作
                action_np = action.squeeze().cpu().numpy()
            eval_state, r, term, trunc, _ = eval_env.step(action_np)
            eval_reward += r
            eval_done = term or trunc
        eval_env.close()

        recent_scores.append(eval_reward)
        all_rewards.append(eval_reward)
        avg_reward = np.mean(recent_scores)
        avg_rewards.append(avg_reward)
        
        if (total_steps // ROLLOUT_STEPS) % 10 == 0:
            print(f"Total Steps: {total_steps}, Episode: {episode_num}, Last Eval Reward: {eval_reward:.2f}, Avg Reward (last 100): {avg_reward:.2f}")
    
    env.close()
    torch.save(ppo_agent.state_dict(), 'ppo_bipedalwalker_icm.pth')

    plt.figure(figsize=(12, 6))
    plt.plot(all_rewards, label='Episode Reward')
    plt.plot(avg_rewards, label='Avg Reward (100 episodes)')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Total Reward')
    plt.title('PPO+ICM Training on BipedalWalker-v3')
    plt.legend()
    plt.savefig('reward_curve_icm.png')
    plt.show()


if __name__ == '__main__':
    train()