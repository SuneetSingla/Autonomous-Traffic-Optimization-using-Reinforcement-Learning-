from simulator import TrafficEnv
from dqn_agent import DQNAgent
from config import *
import torch

env = TrafficEnv()
agent = DQNAgent()

for ep in range(EPISODES):
    state = env.reset()
    total_reward = 0

    for _ in range(MAX_STEPS):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state)
        agent.train()
        state = next_state
        total_reward += reward

    print(f"Episode {ep+1}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

torch.save(agent.model.state_dict(), "models/traffic_dqn.pth")
print("✅ Model saved!")
