# deals with the world state

import numpy as np

class World:
    def __init__(self, cfg):
        self.cfg = cfg
        W, H = cfg.width, cfg.height
        self.food = np.zeros((W, H))
        self.home = np.zeros((W,H))
        self.home_phe = np.zeros((W,H))
        self.food_phe = np.zeros((W,H))
        self.food_delivered = 0
        self.food_patches = [] # for meta data.

        nx, ny = cfg.nest
        self._fill_circle(self.home, nx, ny, cfg.nest_radius, 1.0)

    def add_food(self, x, y, radius, amount_per_cell):
        self._fill_circle(self.food, x, y, radius, amount_per_cell)
        self.food_patches.append(dict(x=x, y=y, radius=radius, 
                                      amount_per_cell = amount_per_cell))

    def update(self):
        # evaporate phe per step
        keep = 1.0 - self.cfg.evaporation
        for grid in (self.food_phe, self.home_phe):
            grid *= keep
            grid[grid < self.cfg.min_phe] = 0.0

    def cell(self, pos):
        x = int(np.clip(pos[0], 0, self.cfg.width - 1))
        y = int(np.clip(pos[1], 0, self.cfg.height - 1))

        return x, y

    def _fill_circle(self, grid, cx, cy, r, value):
        xs, ys = np.ogrid[:grid.shape[0], :grid.shape[1]]
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2
        grid[mask] = value
