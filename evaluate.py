import numpy as np
import matplotlib.pyplot as plt
import torch
from simulator import TrafficEnv
from dqn_agent import DQN

env = TrafficEnv()
model = DQN()
model.load_state_dict(torch.load("models/traffic_dqn.pth"))
model.eval()

def run(policy):
    waits = []
    state = env.reset()
    for _ in range(200):
        if policy == "static":
            action = 0
        else:
            action = torch.argmax(model(torch.FloatTensor(state))).item()
        state, _, _ = env.step(action)
        waits.append(np.sum(env.wait_times))
    return np.mean(waits)

static = run("static")
ai = run("ai")

plt.bar(["Static", "AI"], [static, ai])
plt.ylabel("Average Waiting Time")
plt.title("Traffic Optimization Comparison")
plt.savefig("plots/comparison.png")
plt.show()
