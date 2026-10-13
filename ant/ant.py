import numpy as np

class Ant:
    # Ants own internal state

    def __init__(self, ant_id, pos, rng):
        self.id = ant_id
        self.pos = np.array(pos, dtype=float)
        self.carrying = False
        self.steps_since_event = 0 # steps since last event
        self.last_action = rng.random(4) # Random initialization

    def has_food(self):
        return self.carrying
