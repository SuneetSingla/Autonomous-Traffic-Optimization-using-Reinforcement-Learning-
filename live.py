import matplotlib.pyplot as plt
import numpy as np
import torch
from simulator import TrafficEnv
from dqn_agent import DQN

env = TrafficEnv()
model = DQN()
model.load_state_dict(torch.load("models/traffic_dqn.pth"))
model.eval()

plt.ion()
fig, ax = plt.subplots()

state = env.reset()

for step in range(200):
    ax.clear()

    action = torch.argmax(model(torch.FloatTensor(state))).item()
    state, _, _ = env.step(action)

    lanes = ["North", "South", "East", "West"]
    queues = env.queues

    colors = ["green" if action == 0 else "red"] * 2 + \
             ["green" if action == 1 else "red"] * 2

    ax.bar(lanes, queues, color=colors)
    ax.set_ylim(0, max(queues) + 5)
    ax.set_title("Live Traffic Simulation (AI Controlled)")
    ax.set_ylabel("Number of Vehicles")

    if env.ambulance_lane is not None:
        ax.text(1.5, max(queues),
                f"🚑 Ambulance in {lanes[env.ambulance_lane]} lane",
                ha="center", color="red", fontsize=10)

    plt.pause(0.3)

plt.ioff()
plt.show()
