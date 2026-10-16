from dataclasses import dataclass

@dataclass
class Config:
    # world
    width: int = 100
    height: int = 100
    nest: tuple = (50, 50)
    nest_radius: int = 3

    #ants
    n_ants: int =100 # number of ants
    speed: float = 1.0 # how many steps we can move
    sense_radius: int = 3 # how much we can sense ahead

    #pheromone
    deposit: float = 1.0
    deposit_decay: float = 0.01
    evaporation: float = 0.001
    min_phe: float = 0.01 # below this phe is gone

    #random Walk
    explore: float = 0.25 # chance of wandering off.
    keep_direction: float = 0.75 # pob of keeping the same direction

    # random seed
    seed: int = 0
