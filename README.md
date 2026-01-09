# Autonomous Traffic Optimization using Reinforcement Learning

## Overview
This project implements an AI-based traffic signal controller using Reinforcement Learning (Deep Q-Network) to optimize traffic flow at a 4-way intersection.  
The system learns to minimize vehicle waiting time under varying traffic densities in a simulated environment.

## Simulation Environment
A custom Python-based traffic simulator models a 4-way intersection (North, South, East, West) with stochastic vehicle arrivals.  
Traffic density varies over time, allowing realistic congestion scenarios.

## Reinforcement Learning Setup
- **Algorithm:** Deep Q-Network (DQN)
- **State:** Number of cars and waiting time in each lane
- **Action:** Switch traffic light (North–South or East–West)
- **Reward:** Negative cumulative waiting time of all vehicles

## Adaptive Logic
The system includes emergency handling logic.  
If an ambulance (simulated high-priority vehicle) is detected in any lane, the signal is immediately overridden to clear that lane.

## Evaluation & Analytics
Performance is evaluated by comparing:
- Static fixed-timer signal
- AI-controlled signal

Metrics include average waiting time across different traffic densities.  
Results demonstrate that the AI controller consistently outperforms the static baseline.

## Outputs
- Trained RL model (`.pth`)
- Performance graphs (Static vs AI)
- Live traffic simulation visualization

## Tech Stack
Python, PyTorch, NumPy, Matplotlib
