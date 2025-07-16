# 🧠 Curiosity-Driven PPO Agent for BipedalWalker-v3

This repository implements a Proximal Policy Optimization (PPO) agent enhanced with an Intrinsic Curiosity Module (ICM) to improve sample efficiency and exploration performance in the continuous control environment BipedalWalker-v3 from OpenAI Gym.

# DEMO

![](https://github.com/EricChen0104/curiosity-ppo-bipedalwalker/assets/DEMO_gif.gif)

# 🚀 Features

- Curiosity-Driven Exploration: Encourages the agent to explore novel states using ICM.
- Generalized Advantage Estimation (GAE): For stable and efficient advantage calculation.
- Continuous Action Space with Gaussian Policies: Well-suited for BipedalWalker-v3.
- On-Policy PPO Optimization: With clipping and entropy regularization.
- Realtime Logging and Visualization: Training rewards are plotted and saved.

# 🧩 Architecture

- Actor-Critic Network: Outputs mean and standard deviation of a Gaussian policy.
- ICM Module: Contains inverse and forward models to calculate intrinsic rewards.
- Rollout Buffer: Stores trajectories for batch-based PPO update.

# 📁 Project Structure

```bash

.
├── main.py               # Main training loop
├── actor_critic.py       # ActorCritic neural network
├── icm.py                # Intrinsic Curiosity Module
└── utils.py              # (optional) Helper functions (e.g., plotting, saving)

```

# 🛠️ Installation

```bash

git clone https://github.com/your-username/curiosity-ppo-bipedalwalker.git
cd curiosity-ppo-bipedalwalker
pip install -r requirements.txt

```

# 🧪 Usage

train:

```bash

python main.py

```

test:

```bash

python test.py

```

- Training runs on M1 GPU (Apple Silicon) if available via Metal Performance Shaders (torch.mps)
- Model and reward plots will be saved automatically

# 📊 Results

The combination of PPO and ICM significantly improves early exploration and training stability.

---

| Method | Final Reward | Sample Efficiency |
| PPO Only | ~200 | ❌ Slow convergence |
| PPO + ICM (Ours) | ~300+ | ✅ Faster and stable |

---

# 📖 References

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017, July). Curiosity-driven exploration by self-supervised prediction. In International conference on machine learning (pp. 2778-2787). PMLR.

# ⭐️ Contribute

If you find this project helpful, feel free to:

- ⭐ Star this repository
- 📥 Fork it for your own curiosity research
- 📧 Open an issue or pull request to improve it!
