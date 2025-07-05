import numpy as np
import random
from collections import defaultdict

class RaceCarEnv:
    def __init__(self, track):
        self.track = np.array(track)
        self.height, self.width = self.track.shape
        self.start_positions = [(i, j) for i in range(self.height) for j in range(self.width) if self.track[i, j] == 2]
        self.finish_positions = [(i, j) for i in range(self.height) for j in range(self.width) if self.track[i, j] == 3]
        self.reset()

    def reset(self):
        self.pos = random.choice(self.start_positions)
        self.vel = (0, 0)
        return (*self.pos, *self.vel)
    
    def is_valid(self, i, j):
        return 0 <= i < self.height and 0 <= j < self.width and self.track[i, j] > 0
    
    def get_path(self, pos, vel):
        i0, j0 = pos
        di, dj = vel
        steps = max(abs(di), abs(dj))
        path = [(i0 + round(k * di / steps), j0 + round(k * dj / steps)) for k in range(1, steps + 1)]
        return path
    
    def move(self, pos, vel):
        path = self.get_path(pos, vel)
        for i, j in path:
            if (i, j) in self.finish_positions:
                return (i, j), True, False # pos, finished, crashed
            if not self.is_valid(i, j):
                return random.choice(self.start_positions), False, True # pos, finished, crashed
        return path[-1], False, False # pos, finished, crashed
    
    def step(self, action):
        # Acción aleatoria con probabilidad 0.1
        if random.random() < 0.1:
            action = (0, 0)
        vy, vx = self.vel
        dy, dx = action
        new_vy = min(max(vy + dy, 0), 4)
        new_vx = min(max(vx + dx, 0), 4)
        if new_vy == 0 and new_vx == 0 and self.track[self.pos] != 2:
            new_vy, new_vx = vy, vx # No se permite detenerse fuera de la salida
        new_vel = (new_vy, new_vx)

        new_pos, finished, crashed = self.move(self.pos, new_vel)
        self.pos = new_pos
        self.vel = (0, 0) if crashed else new_vel

        reward = -1
        done = finished
        return (*self.pos, *self.vel), reward, done
    
    def get_all_states(self):
        return [(i, j, vy, vx)
                for i in range(self.height)
                for j in range(self.width)
                if self.track[i, j] > 0
                for vy in range(5)
                for vx in range(5)
                if (vy, vx) != (0, 0) or self.track[i, j] == 2]
    
    def get_actions(self, state):
        return [(dy, dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1]]