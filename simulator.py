import numpy as np
import random

class TrafficEnv:
    def __init__(self, arrival_rate=1):
        self.arrival_rate = arrival_rate
        self.reset()

    def reset(self):
        self.queues = np.random.randint(0, 5, size=4)
        self.wait_times = np.zeros(4)
        self.light = 0
        self.ambulance_lane = None
        return self._get_state()

    def _get_state(self):
        return np.concatenate([self.queues, self.wait_times, [self.light]])

    def step(self, action):
        reward = 0

        if random.random() < 0.02:
            self.ambulance_lane = random.randint(0, 3)
        else:
            self.ambulance_lane = None

        if self.ambulance_lane is not None:
            green_lanes = [self.ambulance_lane]
        else:
            self.light = action
            green_lanes = [0,1] if action == 0 else [2,3]

        for i in range(4):
            arrivals = np.random.poisson(self.arrival_rate)
            self.queues[i] += arrivals

            if i in green_lanes and self.queues[i] > 0:
                self.queues[i] -= 1
            else:
                self.wait_times[i] += self.queues[i]

        reward = -np.sum(self.wait_times)
        done = False
        return self._get_state(), reward, done
