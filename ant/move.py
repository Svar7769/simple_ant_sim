# turns 4 dim action into new position
import numpy as np
from ant.ant_utils import REVERSE

def move(ant, action, world):
    u, d, l, r = action
    arrow = np.array([r - l, u - d])

    length = np.linalg.norm(arrow)
    if length == 0:
        return # for no action

    new_pos = ant.pos + world.cfg.speed * arrow / length

    # if on the edge reverse
    if not (0 <= new_pos[0] < world.cfg.width and 0 <= new_pos[1] < world.cfg.height):
        ant.last_action = action[REVERSE]
        return
    
    ant.pos = new_pos
    ant.last_action = action
