# Read world around one ant (Local observation)
from dataclasses import dataclass
import numpy as np
from ant.ant_utils import DIRECTIONS

@dataclass
class LocalState:
    on_food: bool
    at_home: bool
    see_food: np.ndarray # is there food nearby (direction)
    see_home: np.ndarray # is home nearby (direction)
    smell_food: np.ndarray # nearby food pheromone
    smell_home: np.ndarray # home pheromone in each direction

def look(grid, x, y, radius):
    # looks throught the grids and search for the direction with the most value.
    # Radius controlls the window of look ahead
    W, H = grid.shape
    out = np.zeros(4)
    for d, (dx, dy) in enumerate(DIRECTIONS):
        for step in range(1, radius + 1):
            cx, cy = x + dx * step, y + dy*step
            if 0 <= cx < W and 0 <= cy < H:
                out[d] = max(out[d], grid[cx, cy])
    return out


def sense(ant, world):
    # actually interacts with env and gets all the grids
    x, y = world.cell(ant.pos)
    r = world.cfg.sense_radius
    return LocalState(
        on_food = world.food[x, y] > 0,
        at_home = world.home[x, y] > 0,
        see_food = look(world.food, x, y, r),
        see_home = look(world.home, x, y, r),
        smell_food = look(world.food_phe, x, y, r),
        smell_home = look(world.home_phe, x, y, r),
    )
