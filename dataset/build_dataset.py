# Turn simulated runs into reusable transformer training data.

# use python -m dataset.build_dataset --runs runs/dataset_v0 --out data/brain_v0

import argparse
import glob
import json
import os
import subprocess
import time

import numpy as np

from dataset.config import (
    CFG, DIRS, REVERSE, OBS_LAYOUT, FIELDS, MODES, CASES,
    CARRYING, SEE_FOOD, SEE_HOME, SMELL_FOOD, SMELL_HOME, LAST_ACTION,
    CASE_TARGET, CASE_SMELL, CASE_NOTHING,
)


# recreates the decision of ant brain
def rule_features(obs):
    carrying = obs[:, CARRYING] > 0
    target = np.where(carrying[:, None], obs[:, SEE_HOME], obs[:, SEE_FOOD])
    smell = np.where(carrying[:, None], obs[:, SMELL_HOME], obs[:, SMELL_FOOD]).copy()
    behind = REVERSE[obs[:, LAST_ACTION].argmax(1)]
    smell[np.arange(len(obs)), behind] = 0.0

    has_target = target.sum(1) > 0
    has_smell = smell.sum(1) > 0
    case = np.full(len(obs), CASE_NOTHING, dtype=np.int8)
    case[has_smell] = CASE_SMELL
    case[has_target] = CASE_TARGET

    signal = np.where(has_target[:, None], target, smell)
    n_max = (signal == signal.max(1, keepdims=True)).sum(1)
    tie = (case != CASE_NOTHING) & (n_max > 1)
    return case, behind.astype(np.int8), tie

# Data formating
def run_to_rows(d, run_idx):
    T, N, D = d["obs"].shape
    obs = d["obs"].reshape(-1, D)
    act = d["dec_action"].reshape(-1, 4)
    mode = d["mode"].reshape(-1).astype(np.int8)

    direction = np.where(mode < 2, act.argmax(1), -1).astype(np.int8)
    case, behind, tie = rule_features(obs)

    t = np.repeat(d["t"], N).astype(np.int16)
    ant = np.tile(np.arange(N), T).astype(np.int16)
    food_present = np.repeat(d["food_left"] > 0, N)

    return dict(
        obs=obs.astype(np.float32),
        mode=mode,
        direction=direction,
        pick=d["dec_pick"].reshape(-1),
        drop=d["dec_drop"].reshape(-1),
        case=case,
        behind=behind,
        tie=tie,
        food_present=food_present,
        run=np.full(len(obs), run_idx, dtype=np.int16),
        t=t,
        ant=ant,
    )

# Data check
def check_rows(rows, explore):
    m, c = rows["mode"], rows["case"]
    bad = {
        "target_mode_without_target": int(((m == 0) & (c != CASE_TARGET)).sum()),
        "target_visible_not_target":  int(((c == CASE_TARGET) & (m != 0)).sum()),
        "nothing_but_not_wander":     int(((c == CASE_NOTHING) & (m != 2)).sum()),
        "trail_without_smell":        int(((m == 1) & (c != CASE_SMELL)).sum()),
    }
    smell = c == CASE_SMELL
    p_wander = float((m[smell] == 2).mean()) if smell.any() else float("nan")
    return bad, p_wander

# seed split
def split_of(seed, val_seeds=None, test_seeds=None):
    is_test = seed in test_seeds if test_seeds is not None else seed % 10 == CFG.test_mod
    is_val = seed in val_seeds if val_seeds is not None else seed % 10 == CFG.val_mod
    if is_test:
        return "test"
    if is_val:
        return "val"
    return "train"

# Count rows
def counts(rows):
    n = len(rows["mode"])
    return dict(
        rows=n,
        mode={k: int((rows["mode"] == i).sum()) for i, k in enumerate(MODES)},
        case={k: int((rows["case"] == i).sum()) for i, k in enumerate(CASES)},
        pick=int(rows["pick"].sum()),
        drop=int(rows["drop"].sum()),
        tie=int(rows["tie"].sum()),
        food_present=int(rows["food_present"].sum()),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=CFG.runs_dir)
    ap.add_argument("--out", default=CFG.out_dir)
    ap.add_argument("--val_seeds", type=int, nargs="*", default=CFG.val_seeds)
    ap.add_argument("--test_seeds", type=int, nargs="*", default=CFG.test_seeds)
    args = ap.parse_args()

    run_dirs = sorted(d for d in glob.glob(os.path.join(args.runs, "*/"))
                      if os.path.exists(os.path.join(d, "traj.npz")))
    # error case
    if not run_dirs:
        raise SystemExit(f"no runs with traj.npz found in {args.runs}")


    runs, parts = [], {"train": [], "val": [], "test": []}
    for i, rd in enumerate(run_dirs):
        with open(os.path.join(rd, "meta.json")) as f:
            meta = json.load(f)
        d = np.load(os.path.join(rd, "traj.npz"))
        if "obs" not in d.files:
            raise SystemExit(f"{rd} has no obs field, re-run it with the new collector")
        if meta.get("obs_layout") not in (None, OBS_LAYOUT):
            raise SystemExit(f"{rd}: obs_layout differs from dataset/config.py")

        seed = meta["sim_config"]["seed"]
        split = split_of(seed, args.val_seeds, args.test_seeds)

        rows = run_to_rows(d, i)
        bad, p_wander = check_rows(rows, meta["sim_config"]["explore"])
        if any(bad.values()):
            raise SystemExit(f"{rd}: labels disagree with recomputed rules: {bad}")

        parts[split].append(rows)
        runs.append(dict(
            idx=i,
            name=os.path.basename(os.path.normpath(rd)),
            split=split,
            seed=seed,
            n_ants=meta["n_ants"],
            steps=meta["steps"],
            n_patches=len(meta["food_patches"]),
            food_patches=meta["food_patches"],
            food_gone_at=int(np.argmax(d["food_left"] == 0)) if (d["food_left"] == 0).any() else None,
            p_wander_given_smell=round(p_wander, 4),
            rows=len(rows["mode"]),
        ))
        print(f"{split:5s} {runs[-1]['name']}  rows {runs[-1]['rows']:7d}  "
              f"P(wander|smell) {p_wander:.3f}")

    os.makedirs(args.out, exist_ok=True)
    split_counts = {}
    for split, lst in parts.items():
        if not lst:
            continue
        merged = {k: np.concatenate([r[k] for r in lst]) for k in lst[0]}
        np.savez_compressed(os.path.join(args.out, f"{split}.npz"), **merged)
        split_counts[split] = counts(merged)
        print(f"saved {split}: {split_counts[split]['rows']} rows")

    first_meta = json.load(open(os.path.join(run_dirs[0], "meta.json")))
    manifest = dict(
        version=CFG.version,
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        source_runs_dir=os.path.abspath(args.runs),
        sim_config=first_meta["sim_config"],
        obs_layout=OBS_LAYOUT,
        directions=DIRS,
        fields=FIELDS,
        expected_p_wander={"target": 0.0, "smell": first_meta["sim_config"]["explore"], "nothing": 1.0},
        split_rule="by seed; whole runs go to one split",
        val_seeds=args.val_seeds,
        test_seeds=args.test_seeds,
        counts=split_counts,
        runs=runs,
    )
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote", os.path.join(args.out, "manifest.json"))


if __name__ == "__main__":
    main()
