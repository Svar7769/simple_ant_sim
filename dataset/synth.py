# Synthetic test grid: hand-built brain inputs, labelled by the real brain.decide().
#
#   python -m dataset.synth                  (writes CFG.out_dir/synth.npz, adds "synth" to manifest.json)
#
# Each row belongs to one probe (see PROBES in dataset/config.py) that isolates one rule.
# Rows are labelled by calling ant.brain.decide() itself, so labels follow the brain if you change it.
# The explore coin is forced off when labelling, so for smell rows:
#   mode      = what the ant does when it does NOT explore (trail)
#   p_wander  = true probability of wander for this input (0, explore, or 1)

import argparse
import json
import os
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

from ant.brain import decide
from ant.sense import LocalState
from config import Config as SimConfig
from dataset.build_dataset import rule_features
from dataset.config import (
    CFG, SYNTH, PROBES, REVERSE, OBS_LAYOUT,
    CASE_TARGET, CASE_SMELL, CASE_NOTHING,
)

CHANNELS = ["see_food", "see_home", "smell_food", "smell_home"]


class NoExplore:
    """rng stand-in for decide(): the explore coin never fires (random() = 1.0)."""
    def random(self, size=None):
        return 1.0 if size is None else np.ones(size)


class Gen:
    def __init__(self, scfg, rng):
        self.c, self.rng = scfg, rng

    # ---------- building blocks ----------
    def blank(self, carrying):
        row = {k: np.zeros(4) for k in CHANNELS}
        row.update(on_food=False, at_home=False, carrying=bool(carrying))
        heading = int(self.rng.integers(4))
        row["last_action"] = self.last_action(heading)
        row["behind"] = int(REVERSE[heading])
        return row

    def last_action(self, heading):
        if self.rng.random() < self.c.onehot_last_action_frac:
            v = np.zeros(4)
            v[heading] = 1.0
            return v
        v = self.rng.random(4) * 0.9          # continuous, wander style
        v[heading] = 0.95 + 0.05 * self.rng.random()
        return v

    @staticmethod
    def rel(row):
        """(target, smell) channel names the brain uses for this carrying state."""
        return ("see_home", "smell_home") if row["carrying"] else ("see_food", "smell_food")

    @staticmethod
    def irr(row):
        return ("see_food", "smell_food") if row["carrying"] else ("see_home", "smell_home")

    def target_val(self, channel):
        lo, hi = self.c.food_values
        return 1.0 if channel == "see_home" else float(self.rng.integers(lo, hi + 1))

    def smell_val(self, lo=None, hi=None):
        a, b = self.c.smell_range
        return float(self.rng.uniform(lo if lo is not None else a, hi if hi is not None else b))

    def distract(self, row, p=0.5):
        """Random values in the channels the brain should ignore."""
        t, s = self.irr(row)
        for d in range(4):
            if self.rng.random() < p:
                row[t][d] = self.target_val(t)
            if self.rng.random() < p:
                row[s][d] = self.smell_val()

    def ahead(self, row, k):
        """k distinct directions that are not behind."""
        dirs = [d for d in range(4) if d != row["behind"]]
        return list(self.rng.permutation(dirs)[:k])

    def descending_smells(self, k):
        """k distinct smell values, largest first, max - second >= min_gap."""
        a, b = self.c.smell_range
        g = self.c.min_gap
        vals = [self.smell_val(a + g * (k - 1), b)]
        for rem in range(k - 2, -1, -1):                 # leave room for the values still to come
            vals.append(self.smell_val(a + g * rem, vals[-1] - g))
        return vals

    # ---------- probes: each returns (row, expected) ----------
    def gate_target(self):
        row = self.blank(self.rng.random() < 0.5)
        t, _ = self.rel(row)
        ti, _ = self.irr(row)
        a, b = self.rng.permutation(4)[:2]
        row[t][a] = self.target_val(t)
        row[ti][b] = self.target_val(ti)
        return row, dict(mode=0, direction=int(a))

    def gate_smell(self):
        row = self.blank(self.rng.random() < 0.)
        _, s = self.rel(row)
        _, si = self.irr(row)
        a = self.ahead(row, 1)[0]
        b = int(self.rng.choice([d for d in range(4) if d != a]))
        lo, hi = self.c.smell_range
        row[s][a] = self.smell_val(lo, hi / 2)
        row[si][b] = self.smell_val(hi / 2, hi)          # stronger, but wrong channel
        return row, dict(mode=1, direction=int(a))

    def priority(self):
        row = self.blank(self.rng.random() < 0.5)
        t, s = self.rel(row)
        a = int(self.rng.integers(4))
        b = self.ahead(row, 3)
        b = int([d for d in b if d != a][0])
        row[t][a] = self.target_val(t)
        row[s][b] = self.smell_val(0.5, 1.0)
        self.distract(row)
        return row, dict(mode=0, direction=a)

    def argmax_target(self):
        row = self.blank(False)                          # food values differ; home is always 1.0
        k = int(self.rng.integers(2, 5))
        dirs = self.rng.permutation(4)[:k]
        lo, hi = self.c.food_values
        vals = self.rng.choice(np.arange(lo, hi + 1), size=k, replace=False).astype(float)
        row["see_food"][dirs] = vals
        row["smell_food"][self.rng.integers(4)] = self.smell_val()
        self.distract(row)
        return row, dict(mode=0, direction=int(dirs[np.argmax(vals)]))

    def argmax_smell(self):
        row = self.blank(self.rng.random() < 0.5)
        _, s = self.rel(row)
        k = int(self.rng.integers(2, 4))
        dirs = self.ahead(row, k)
        vals = self.descending_smells(k)
        for d, v in zip(dirs, vals):
            row[s][d] = v
        if self.rng.random() < 0.5:                      # weaker smell behind, should not matter
            row[s][row["behind"]] = self.smell_val(None, vals[-1])
        self.distract(row)
        return row, dict(mode=1, direction=int(dirs[0]))

    def behind(self):
        row = self.blank(self.rng.random() < 0.5)
        _, s = self.rel(row)
        k = int(self.rng.integers(1, 4))
        dirs = self.ahead(row, k)
        vals = self.descending_smells(k)
        for d, v in zip(dirs, vals):
            row[s][d] = v
        row[s][row["behind"]] = self.smell_val(min(vals[0] + self.c.min_gap, 0.999), 1.0)  # strongest is behind
        self.distract(row)
        return row, dict(mode=1, direction=int(dirs[0]))

    def tie(self):
        row = self.blank(self.rng.random() < 0.5)
        t, s = self.rel(row)
        if self.rng.random() < 0.5:                      # tie in targets (behind does not mask targets)
            dirs = self.rng.permutation(4)[:2]
            v = self.target_val(t)
            row[t][dirs] = v
            mode = 0
        else:                                            # tie in smell ahead
            dirs = np.array(self.ahead(row, 2))
            v = self.smell_val(0.1, 1.0)
            row[s][dirs] = v
            if self.rng.random() < 0.5:
                row[s][row["behind"]] = self.smell_val(None, v)
            mode = 1
        self.distract(row)
        return row, dict(mode=mode, direction=int(dirs.min()))   # np.argmax -> first index

    def nothing(self):
        row = self.blank(self.rng.random() < 0.5)
        _, s = self.rel(row)
        if self.rng.random() < 0.5:                      # smell only behind
            row[s][row["behind"]] = self.smell_val()
        self.distract(row, p=0.7)
        return row, dict(mode=2)

    def pick_drop(self):
        row = self.blank(self.rng.random() < 0.5)
        row["on_food"] = bool(self.rng.random() < 0.5)
        row["at_home"] = bool(self.rng.random() < 0.5)
        if row["on_food"]:
            for d in self.rng.permutation(4)[: int(self.rng.integers(1, 5))]:
                row["see_food"][d] = self.target_val("see_food")
        if row["at_home"]:
            for d in self.rng.permutation(4)[: int(self.rng.integers(1, 5))]:
                row["see_home"][d] = 1.0
        return row, dict(pick=row["on_food"] and not row["carrying"],
                         drop=row["at_home"] and row["carrying"])


def to_obs(row):
    return np.concatenate([
        [row["on_food"], row["at_home"], row["carrying"]],
        row["see_food"], row["see_home"], row["smell_food"], row["smell_home"],
        row["last_action"],
    ]).astype(np.float32)


def label(row, sim_cfg):
    ant = SimpleNamespace(last_action=row["last_action"].copy(),
                          has_food=lambda c=row["carrying"]: c)
    local = LocalState(on_food=row["on_food"], at_home=row["at_home"],
                       see_food=row["see_food"], see_home=row["see_home"],
                       smell_food=row["smell_food"], smell_home=row["smell_home"])
    dec = decide(ant, local, NoExplore(), sim_cfg)
    direction = int(np.argmax(dec.action)) if dec.mode < 2 else -1
    return dec.mode, direction, bool(dec.pick), bool(dec.drop)


def build(scfg=SYNTH, sim_cfg=None):
    sim_cfg = sim_cfg or SimConfig()
    rng = np.random.default_rng(scfg.seed)
    gen = Gen(scfg, rng)

    cols = {k: [] for k in ["obs", "mode", "direction", "pick", "drop", "probe"]}
    mismatches = {}
    for p_idx, name in enumerate(PROBES):
        bad = 0
        for _ in range(scfg.n_per_probe):
            row, exp = getattr(gen, name)()
            mode, direction, pick, drop = label(row, sim_cfg)
            got = dict(mode=mode, direction=direction, pick=pick, drop=drop)
            bad += any(got[k] != v for k, v in exp.items())
            cols["obs"].append(to_obs(row))
            cols["mode"].append(mode)
            cols["direction"].append(direction)
            cols["pick"].append(pick)
            cols["drop"].append(drop)
            cols["probe"].append(p_idx)
        mismatches[name] = bad

    obs = np.stack(cols["obs"])
    case, behind, tie = rule_features(obs)
    mode = np.array(cols["mode"], dtype=np.int8)
    p_wander = np.select([case == CASE_TARGET, case == CASE_SMELL, case == CASE_NOTHING],
                         [0.0, sim_cfg.explore, 1.0]).astype(np.float32)
    n = len(obs)
    arrays = dict(
        obs=obs,
        mode=mode,
        direction=np.array(cols["direction"], dtype=np.int8),
        pick=np.array(cols["pick"], dtype=bool),
        drop=np.array(cols["drop"], dtype=bool),
        case=case, behind=behind, tie=tie,
        probe=np.array(cols["probe"], dtype=np.int8),
        p_wander=p_wander,
        food_present=np.ones(n, dtype=bool),        # same fields as real splits
        run=np.full(n, -1, dtype=np.int16),
        t=np.zeros(n, dtype=np.int16),
        ant=np.zeros(n, dtype=np.int16),
    )

    # consistency: decide() (explore off) must agree with the rules recomputed from obs
    consistency = {
        "mode_vs_case": int((mode != case).sum()),            # 0 target, 1 trail, 2 wander
        "tie_probe_without_tie": int((~tie[arrays["probe"] == PROBES.index("tie")]).sum()),
    }
    return arrays, mismatches, consistency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=CFG.out_dir)
    ap.add_argument("--n_per_probe", type=int, default=SYNTH.n_per_probe)
    ap.add_argument("--seed", type=int, default=SYNTH.seed)
    args = ap.parse_args()

    scfg = SYNTH.__class__(**{**asdict(SYNTH), "n_per_probe": args.n_per_probe, "seed": args.seed})
    arrays, mismatches, consistency = build(scfg)

    for name in PROBES:
        print(f"{name:14s} rows {scfg.n_per_probe:6d}  mismatches vs expected {mismatches[name]}")
    print("consistency", consistency)
    if any(mismatches.values()) or any(consistency.values()):
        raise SystemExit("synthetic labels disagree with expectations: brain.py changed or synth bug")

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "synth.npz"), **arrays)

    man_path = os.path.join(args.out, "manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else dict(obs_layout=OBS_LAYOUT)
    manifest["synth"] = dict(
        config=asdict(scfg),
        probes=PROBES,
        rows=int(len(arrays["mode"])),
        rows_per_probe=scfg.n_per_probe,
        tie_rows=int(arrays["tie"].sum()),
        note="mode/direction labelled with explore coin off; use p_wander for calibration",
        extra_fields={"probe": "int8 index into probes", "p_wander": "float32 true P(wander)"},
    )
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("saved", os.path.join(args.out, "synth.npz"), "and updated manifest.json")


if __name__ == "__main__":
    main()
