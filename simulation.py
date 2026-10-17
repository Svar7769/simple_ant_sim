# Loop to run simulation

import numpy as np
from env.world import World
from ant.ant import Ant
from ant.sense import sense
from ant.brain import decide
from ant.move import move
from ant.ant_utils import pick_food, drop_food, drop_pheromone
from ant.ant_utils import EVENT_NONE, EVENT_PICK, EVENT_DROP

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
        
        ant.mode = decision.mode
        ant.event = EVENT_NONE

        #act
        was_carrying = ant.carrying 
        if decision.pick:
            pick_food(ant, self.world)
        elif decision.drop:
            drop_food(ant, self.world)

        # Event Log
        if ant.carrying and not was_carrying:
            ant.event = EVENT_PICK
        elif was_carrying and not ant.carrying:
            ant.event = EVENT_DROP

        # Move ant
        move(ant, decision.action, self.world)
        # mark
        drop_pheromone(ant, self.world)
        # increment clock
        ant.steps_since_event += 1

        # Saving data for the transformer
        ant.obs = np.concatenate([
            [local.on_food, local.at_home, ant.carrying],
            local.see_food, local.see_home,
            local.smell_food, local.smell_home,
            ant.last_action,
        ]).astype(np.float32)
        ant.dec_action = np.asarray(decision.action, dtype=np.float32).copy()
        ant.dec_pick = bool(decision.pick)
        ant.dec_drop = bool(decision.drop)

    def state(self):
        pos = np.array([a.pos for a in self.ants])
        carrying = np.array([a.carrying for a in self.ants])
        return pos, carrying
