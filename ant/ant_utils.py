# shared utilities

import numpy as np

DIRECTIONS = [(0, 1), (0, -1), (-1, 0), (1, 0)] #up, down, left, right
REVERSE = [1, 0, 3, 2]   # up<->down, left<->right

# move 360 degree
def turn_around(ant):
    ant.last_action = ant.last_action[REVERSE]

# Pick food
def pick_food(ant, world):
    x, y = world.cell(ant.pos)
    if world.food[x,y] > 0:
        world.food[x,y] -= 1
        ant.carrying = True
        ant.steps_since_event = 0
        turn_around(ant)

# drop food
def drop_food(ant, world):
    world.food_delivered += 1
    ant.carrying = False
    ant.steps_since_event = 0
    turn_around(ant)

# drop a pheromone
def drop_pheromone(ant, world):
    cfg = world.cfg
    strength = cfg.deposit * np.exp(-cfg.deposit_decay * ant.steps_since_event)
    x, y = world.cell(ant.pos)
    grid = world.food_phe if ant.has_food() else world.home_phe
    grid[x, y] = max(grid[x, y], strength)
