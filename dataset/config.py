# One place for everything the dataset pipeline needs.
# build_dataset.py and ant_dataset.py import from here; change values here only.

from dataclasses import dataclass, field

import numpy as np

# layout (must match Simulation.step_ant)
DIRS = ["up", "down", "left", "right"]
REVERSE = np.array([1, 0, 3, 2])          # up<->down, left<->right

OBS_LAYOUT = [
    "on_food", "at_home", "carrying",
    "see_food_u", "see_food_d", "see_food_l", "see_food_r",
    "see_home_u", "see_home_d", "see_home_l", "see_home_r",
    "smell_food_u", "smell_food_d", "smell_food_l", "smell_food_r",
    "smell_home_u", "smell_home_d", "smell_home_l", "smell_home_r",
    "last_action_u", "last_action_d", "last_action_l", "last_action_r",
]
OBS_DIM = len(OBS_LAYOUT)

ON_FOOD, AT_HOME, CARRYING = 0, 1, 2
SEE_FOOD, SEE_HOME = slice(3, 7), slice(7, 11)
SMELL_FOOD, SMELL_HOME = slice(11, 15), slice(15, 19)
LAST_ACTION = slice(19, 23)

# ---------------- labels ----------------
MODES = ["target", "trail", "wander"]

# behavioural case
CASE_TARGET = 0     # target visible
CASE_SMELL = 1      # smell ahead, no target
CASE_NOTHING = 2    # nothing ahead
CASES = ["target", "smell", "nothing"]

FIELDS = {
    "obs":       "float32 (R,23) brain input, layout in manifest.obs_layout",
    "mode":      "int8 0 target / 1 trail / 2 wander",
    "direction": "int8 0..3 (up,down,left,right) for mode 0/1, -1 for wander",
    "pick":      "bool decision.pick",
    "drop":      "bool decision.drop",
    "case":      "int8 0 target visible / 1 smell only / 2 nothing",
    "behind":    "int8 direction masked by ignore_behind = REVERSE[argmax(last_action)]",
    "tie":       "bool relevant signal (target if case 0, masked smell if case 1) has a tied max",
    "food_present": "bool food left in world at this step",
    "run":       "int16 index into manifest.runs",
    "t":         "int16 sim step",
    "ant":       "int16 ant id",
}

# model inputs
DIR_FEATURES = ["see_food", "see_home", "smell_food", "smell_home", "last_action"]
SELF_FEATURES = ["on_food", "at_home", "carrying"]
LABELS = ["mode", "direction", "pick", "drop"]
INFO = ["case", "behind", "tie", "food_present", "run", "t", "ant"]

PROBES = [
    "gate_target",    # target in carried-relevant channel vs other channel   -> follow relevant
    "gate_smell",     # same for smell                                         -> follow relevant
    "priority",       # target in dir a, stronger smell in dir b               -> target wins
    "argmax_target",  # several distinct food targets                          -> largest
    "argmax_smell",   # several distinct smells (none behind)                  -> largest
    "behind",         # strongest smell is behind                              -> second strongest
    "tie",            # two equal maxima                                       -> first index (up,down,left,right)
    "nothing",        # nothing ahead (smell only behind or none)              -> wander
    "pick_drop",      # on_food / at_home x carrying                           -> pick, drop rules
]

@dataclass
class DatasetConfig:
    version: str = "brain_v0"
    runs_dir: str = "runs/dataset_v0"
    out_root: str = "data"

    # splits by seed (whole runs).
    val_seeds: list = field(default_factory=lambda: [4, 8, 18])
    test_seeds: list = field(default_factory=lambda: [9, 13, 17])
    val_mod: int = 8                        # used only if val_seeds is None
    test_mod: int = 9                       # used only if test_seeds is None

    # balanced sampler: draw prob for [case target, case smell, case nothing, pick/drop rows]
    group_frac: tuple = (0.25, 0.25, 0.25, 0.25)
    batch_size: int = 4096

    @property
    def out_dir(self):
        return f"{self.out_root}/{self.version}"


CFG = DatasetConfig()

@dataclass
class SynthConfig:
    n_per_probe: int = 5000
    seed: int = 0
    onehot_last_action_frac: float = 0.7    # rest are continuous wander-style vectors
    food_values: tuple = (1, 6)             # see_food reads food per cell (integers)
    smell_range: tuple = (0.01, 1.0)        # pheromone strengths
    min_gap: float = 0.02                   # smallest gap between max and runner-up (no accidental ties)


SYNTH = SynthConfig()
