# Runs the simulation

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from simulation import Simulation

def make_sim(cfg):
    sim = Simulation(cfg)
    sim.world.add_food(80, 80, radius=4, amount_per_cell = 5)
    sim.world.add_food(20, 75, radius=4, amount_per_cell = 5)
    return sim

def draw(ax, sim):
    w = sim.world
    ax.clear()

    # background for each grid
    # red = food
    # blue = home
    # green = food
    img = np.zeros((w.cfg.height, w.cfg.width, 3))
    img[..., 0] = np.clip(w.food_phe.T, 0, 1)
    img[..., 2] = np.clip(w.home_phe.T, 0, 1)
    img[..., 1] = np.clip(w.food.T, 0, 1)
    ax.imshow(img, origin="lower")

    # nest ouline
    nx, ny = w.cfg.nest
    ax.add_patch(plt.Circle((nx, ny), w.cfg.nest_radius, fill=False, color="white"))

    # ants 
    # white -> searching food
    # yellow -> has food going home
    pos, carrying = sim.state()
    ax.scatter(pos[~carrying, 0], pos[~carrying, 1], s=6, c="white")
    ax.scatter(pos[carrying, 0], pos[carrying, 1], s=6, c="yellow")

    ax.set_title(f"step {sim.t}   delivered {w.food_delivered}")
    ax.set_xticks([])
    ax.set_yticks([])

# WE MIGHT WANT TO IMPOVE THIS TO GET MORE DATA. BASED ON OUR NEED
def save(path, position, carrying, cfg):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        pos=np.array(position),        # (T, N, 2)
        carrying=np.array(carrying),    # (T, N)
        seed=cfg.seed,
    )
    print("saved", path + ".npz")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--draw_every", type=int, default=10)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    cfg = Config(seed=args.seed)
    sim = make_sim(cfg)

    if not args.headless:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6,6))

    positions, carrying = [], []

    for t in range(args.steps):
        sim.step()

        if args.save:
            pos, carry = sim.state()
            positions.append(pos)
            carrying.append(carry)

        if args.headless:
            if t % 500 == 0:
                print(f"step {t}  delivered {sim.world.food_delivered}")
        elif t % args.draw_every == 0:
            draw(ax, sim)
            plt.pause(0.001)
            if not plt.fignum_exists(fig.number):
                break

    print(f"Done! food delivered: {sim.world.food_delivered}")

    if args.save:
        save(args.save, positions, carrying, cfg)

    if not args.headless:
        plt.close(fig)
    
if __name__ == "__main__":
    main()
