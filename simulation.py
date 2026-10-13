# Loop to run simulation

import numpy as np
from env.world import World
from ant.ant import Ant
from ant.sense import sense
from ant.brain import decide
from ant.move import move
from ant.ant_utils import pick_food, drop_food, drop_pheromone

class Simulation:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.world = World(cfg)
        self.ants = [Ant(i, cfg.nest, self.rng) for i in range(cfg.n_ants)]
        self.t = 0

    def step(self):
        for ant in self.ants:
            self.step_ant(ant)
        self.world.update()
        self.t += 1

    def step_ant(self, ant):
        # sense
        local = sense(ant, self.world)
        #decide
        decision = decide(ant, local, self.rng, self.cfg)
        #act
        if decision.pick:
            pick_food(ant, self.world)
        elif decision.drop:
            drop_food(ant, self.world)
        # Move ant
        move(ant, decision.action, self.world)
        # mark
        drop_pheromone(ant, self.world)
        # increment clock
        ant.steps_since_event += 1

    def state(self):
        pos = np.array([a.pos for a in self.ants])
        carrying = np.array([a.carrying for a in self.ants])
        return pos, carrying
