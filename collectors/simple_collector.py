#Record Simulation run and saves it!

import json
import os
import subprocess
import time
from dataclasses import asdict

import numpy as np

class SimpleCollector:
    def __init__(self, cfg):
        self.cfg = cfg

        # for ant state
        self.t = []
        self.pos = []
        self.action = []
        self.carrying = []
        self.mode = []
        self.event = []

        # colony stats
        self.delivered = []
        self.food_left = []

        # grid snapshot
        self.snap_t = []
        self.food_phe = []
        self.home_phe = []
        self.food = []

        # Internal Data
        self.obs = []
        self.dec_action = []
        self.dec_pick = []
        self.dec_drop = []
    
    # after every simulation step
    def record(self, sim):
        t = sim.t

        if t % self.cfg.collect_every == 0:
            ants = sim.ants
            self.t.append(t)
            self.pos.append([a.pos for a in ants])
            self.action.append([a.last_action for a in ants])
            self.carrying.append([a.carrying for a in ants])
            self.mode.append([a.mode for a in ants])
            self.event.append([a.event for a in ants])
            self.delivered.append(sim.world.food_delivered)
            self.food_left.append(sim.world.food.sum())
            self.obs.append([a.obs for a in ants])
            self.dec_action.append([a.dec_action for a in ants])
            self.dec_pick.append([a.dec_pick for a in ants])
            self.dec_drop.append([a.dec_drop for a in ants])

        if self.cfg.save_grids and t % self.cfg.snapshot_every == 0:
            w = sim.world
            self.snap_t.append(t)
            self.food_phe.append(w.food_phe.copy())
            self.home_phe.append(w.home_phe.copy())
            self.food.append(w.food.copy())

    def save(self, sim):
        run_dir = os.path.join(self.cfg.out_dir, self._run_name(sim))
        os.makedirs(run_dir, exist_ok=True)

        data = dict(
            t = np.array(self.t, dtype=np.int32),
            pos = np.array(self.pos, dtype = np.float32),
            action = np.array(self.action, dtype=np.float32),
            carrying = np.array(self.carrying, dtype=bool),
            mode = np.array(self.mode, dtype=np.int8),
            event = np.array(self.event, dtype=np.int8),
            delivered = np.array(self.delivered, dtype=np.int32),
            food_left = np.array(self.food_left, dtype=np.float32),
            obs = np.array(self.obs, dtype=np.float32),                # (T, N, 23)
            dec_action = np.array(self.dec_action, dtype=np.float32),  # (T, N, 4)
            dec_pick = np.array(self.dec_pick, dtype=bool),            # (T, N)
            dec_drop = np.array(self.dec_drop, dtype=bool),            # (T, N)
        )

        if self.cfg.save_grids:
            data.update(
                    snap_t = np.array(self.snap_t, dtype=np.int32),
                    food_phe = np.array(self.food_phe, dtype=np.float16),
                    home_phe = np.array(self.home_phe, dtype=np.float16),
                    food = np.array(self.food, dtype=np.float16),
                )
        np.savez_compressed(os.path.join(run_dir, "traj.npz"), **data)

        meta = dict(
                sim_config=asdict(sim.cfg),
                collector_config=asdict(self.cfg),
                food_patches=sim.world.food_patches,
                steps=sim.t,
                n_ants=len(sim.ants),
                created=time.strftime("%Y-%m-%d %H:%M:%S"),
                modes={"0": "target", "1": "trail", "2": "wander"},
                events={"0": "none", "1": "pick", "2": "drop"},
                obs_layout=["on_food", "at_home", "carrying",
                            "see_food_u", "see_food_d", "see_food_l", "see_food_r",
                            "see_home_u", "see_home_d", "see_home_l", "see_home_r",
                            "smell_food_u", "smell_food_d", "smell_food_l", "smell_food_r",
                            "smell_home_u", "smell_home_d", "smell_home_l", "smell_home_r",
                            "last_action_u", "last_action_d", "last_action_l", "last_action_r"],
            )
        
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=3)

        print("saved", run_dir)
        return run_dir

    def _run_name(self, sim):
        if self.cfg.run_name:
            return self.cfg.run_name
        return time.strftime("%Y%m%d_%H%M%S") + f"_seed{sim.cfg.seed}"

# returns (datam meta_data)
def load_run(run_dir):
    d = np.load(os.path.join(run_dir, "traj.npz"))
    data = {k: d[k] for k in d.files}
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    return data, meta
