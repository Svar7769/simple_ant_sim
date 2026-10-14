# Decides what do based on the local state.
# Does not interact with env. We can change this to make more informed or smart decision

from dataclasses import dataclass
import numpy as np
from ant.ant_utils import REVERSE
from ant.ant_utils import MODE_TARGET, MODE_TRAIL, MODE_WANDER
@dataclass
class Decision:
    action: np.ndarray #[up, down, left, rigt]
    pick: bool
    drop: bool
    mode: int


# Simple decision tree
def decide(ant, local, rng, cfg):
    if not ant.has_food():
        # Search Mode : look for food
        action, mode = choose_direction(
                local.see_food, 
                local.smell_food, 
                ant.last_action, 
                rng, 
                cfg
            )
        return Decision(action, pick=local.on_food, drop=False, mode=mode)
    # Carry Mode: Has food go home
    action, mode = choose_direction(
            local.see_home,
            local.smell_home,
            ant.last_action,
            rng,
            cfg
        )
    return Decision(action, pick=False, drop=local.at_home, mode=mode)

# Key function that actually decides what to do.
def choose_direction(target, smell, last_action, rng, cfg):
    if target.sum() > 0:
        return toward(target), MODE_TARGET
    if rng.random() < cfg.explore:
        return wander(last_action, rng, cfg.keep_direction), MODE_WANDER
    smell = ignore_behind(smell, last_action)
    if smell.sum() > 0:
        return toward(smell), MODE_TRAIL
    return wander(last_action, rng, cfg.keep_direction), MODE_WANDER

# move towards strongest smell
def toward(signal):
    action = np.zeros(4)
    action[np.argmax(signal)] = 1.0
    return action

# Don't move back to smell last location is 0
def ignore_behind(signal, last_action):
    signal = signal.copy()
    signal[REVERSE[np.argmax(last_action)]] = 0.0
    return signal

# move randomly
def wander(last_action, rng, keep_direction):
    if rng.random() < keep_direction:
        return last_action
    return rng.random(4)
