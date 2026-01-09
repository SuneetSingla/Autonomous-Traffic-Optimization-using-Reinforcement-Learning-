import numpy as np
import matplotlib.pyplot as plt
import torch
from simulator import TrafficEnv
from dqn_agent import DQN

model = DQN()
model.load_state_dict(torch.load("models/traffic_dqn.pth"))
model.eval()

densities = [0.5, 1.0, 2.0, 3.0]
static_waits = []
ai_waits = []

def run(env, policy):
    state = env.reset()
    for _ in range(200):
        if policy == "static":
            action = 0
        else:
            action = torch.argmax(model(torch.FloatTensor(state))).item()
        state, _, _ = env.step(action)
    return np.sum(env.wait_times)

for d in densities:
    env = TrafficEnv(arrival_rate=d)

    static_waits.append(run(env, "static"))
    ai_waits.append(run(env, "ai"))

plt.plot(densities, static_waits, marker='o', label="Static Signal")
plt.plot(densities, ai_waits, marker='o', label="AI Signal")
plt.xlabel("Traffic Density (Arrival Rate)")
plt.ylabel("Total Waiting Time")
plt.title("Traffic Density Sweep: Static vs AI")
plt.legend()
plt.savefig("plots/density_sweep.png")
plt.show()
