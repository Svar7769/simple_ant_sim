# Runs the simulation

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from simulation import Simulation
from collectors.config import CollectorConfig
from collectors.simple_collector import SimpleCollector


def make_sim(cfg, random_food=False, n_patches=2, food_seed=None):
    sim = Simulation(cfg)
    if not random_food:
        sim.world.add_food(80, 80, radius=4, amount_per_cell = 5)
        sim.world.add_food(20, 75, radius=4, amount_per_cell = 5)
        return sim

    # separate rng so food layout does not change the ants' random stream
    rng = np.random.default_rng(cfg.seed + 10_000 if food_seed is None else food_seed)
    nx, ny = cfg.nest
    placed = []
    tries = 0
    while len(placed) < n_patches and tries < 1000:
        tries += 1
        r = int(rng.integers(3, 6))                  # radius 3..5
        x = int(rng.integers(r + 1, cfg.width - r - 1))
        y = int(rng.integers(r + 1, cfg.height - r - 1))
        if np.hypot(x - nx, y - ny) < 20:            # not too close to the nest
            continue
        if any(np.hypot(x - px, y - py) < r + pr + 4 for px, py, pr in placed):
            continue                                 # no overlapping patches
        amount = int(rng.integers(3, 7))             # 3..6 food per cell
        sim.world.add_food(x, y, radius=r, amount_per_cell=amount)
        placed.append((x, y, r))
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
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--draw_every", type=int, default=10)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="runs")
    parser.add_argument("--n_ants", type=int, default=100)
    parser.add_argument("--random_food", action="store_true")
    parser.add_argument("--n_patches", type=int, default=2)
    args = parser.parse_args()

    cfg = Config(seed=args.seed, n_ants=args.n_ants)
    sim = make_sim(cfg, random_food=args.random_food, n_patches=args.n_patches)

    collector = SimpleCollector(CollectorConfig(out_dir=args.out_dir, run_name=args.name))

    if not args.headless:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6,6))

    positions, carrying = [], []

    for t in range(args.steps):
        sim.step()
        collector.record(sim)

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
    
    collector.save(sim)
    print(f"Done! food delivered: {sim.world.food_delivered}")

    if args.save:
        save(args.save, positions, carrying, cfg)

    if not args.headless:
        plt.close(fig)
    
if __name__ == "__main__":
    main()
