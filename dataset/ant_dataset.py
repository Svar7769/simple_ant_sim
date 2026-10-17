# Uses build dataset to process the runs and then combines them into test, train, val for final ant dataset

import json
import os

import numpy as np

from dataset.config import CFG, DIR_FEATURES, SELF_FEATURES, LABELS, INFO

# Load ant brain data.
class BrainData:
    def __init__(self, root=None, split="train", arrays=None, manifest=None):
        root = root or CFG.out_dir
        self.root, self.split = root, split
        if manifest is None:
            with open(os.path.join(root, "manifest.json")) as f:
                manifest = json.load(f)
        self.manifest = manifest
        if arrays is None:
            d = np.load(os.path.join(root, f"{split}.npz"))
            arrays = {k: d[k] for k in d.files}
        self.a = arrays

    def __len__(self):
        return len(self.a["mode"])

    def __getattr__(self, k):
        a = self.__dict__.get("a", {})
        if k in a:
            return a[k]
        raise AttributeError(k)

    
    def subset(self, mask):
        return BrainData(self.root, self.split, {k: v[mask] for k, v in self.a.items()}, self.manifest)

    # value fillter
    def where(self, **conds):
        mask = np.ones(len(self), dtype=bool)
        for k, v in conds.items():
            mask &= np.isin(self.a[k], np.atleast_1d(v))
        return self.subset(mask)

    def tokens(self):
        """dir_tok: (R,4,5) one token per direction (up,down,left,right) with DIR_FEATURES.
           self_tok: (R,3) SELF_FEATURES."""
        obs = self.a["obs"]
        dir_tok = obs[:, 3:23].reshape(-1, 5, 4).transpose(0, 2, 1).copy()
        self_tok = obs[:, 0:3].copy()
        return dict(dir_tok=dir_tok, self_tok=self_tok)

    def flat(self):
        """(R,23) raw obs, for an MLP baseline."""
        return self.a["obs"]

    def labels(self):
        return {k: self.a[k] for k in LABELS}

        def sample_weights(self, group_frac=None):
        """Per-row weights for a weighted sampler.
        Groups: case 0 (target), case 1 (smell), case 2 (nothing), 3 (pick/drop rows).
        Each group is drawn with probability group_frac.

        Balances by CASE only, never by mode: inside case 1 the trail/wander
        split must stay at its natural 0.75/0.25, or the model learns a wrong
        explore rate."""
        group_frac = CFG.group_frac if group_frac is None else group_frac
        group = self.a["case"].astype(np.int64).copy()
        group[self.a["pick"] | self.a["drop"]] = 3
        cnt = np.bincount(group, minlength=4).astype(float)
        frac = np.asarray(group_frac, dtype=float)
        frac = np.where(cnt > 0, frac, 0.0)
        frac /= frac.sum()
        w = frac[group] / np.maximum(cnt[group], 1)
        return w / w.sum()

    def summary(self):
        m, c = self.a["mode"], self.a["case"]
        return dict(
            split=self.split, rows=len(self),
            mode=np.bincount(m, minlength=3).tolist(),
            case=np.bincount(c, minlength=3).tolist(),
            pick=int(self.a["pick"].sum()), drop=int(self.a["drop"].sum()),
        )

    # ---------- torch ----------
    def to_torch(self, device="cpu"):
        return TorchBrainData(self, device)


try:
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

    class TorchBrainData(Dataset):
        """Whole split as tensors; __getitem__ returns a dict of inputs, labels, info."""

        def __init__(self, data, device="cpu"):
            tok = data.tokens()
            self.t = {
                "dir_tok": torch.as_tensor(tok["dir_tok"]),
                "self_tok": torch.as_tensor(tok["self_tok"]),
                "mode": torch.as_tensor(data.mode.astype(np.int64)),
                "direction": torch.as_tensor(data.direction.astype(np.int64)),   # -1 = ignore
                "pick": torch.as_tensor(data.pick.astype(np.float32)),
                "drop": torch.as_tensor(data.drop.astype(np.float32)),
                "case": torch.as_tensor(data.case.astype(np.int64)),
                "tie": torch.as_tensor(data.tie),
            }
            self.t = {k: v.to(device) for k, v in self.t.items()}

        def __len__(self):
            return len(self.t["mode"])

        def __getitem__(self, i):
            return {k: v[i] for k, v in self.t.items()}

    def make_loader(data, batch_size=None, balanced=False, shuffle=True, num_samples=None, **kw):
        batch_size = batch_size or CFG.batch_size
        ds = data.to_torch()
        if balanced:
            w = torch.as_tensor(data.sample_weights(), dtype=torch.double)
            sampler = WeightedRandomSampler(w, num_samples or len(ds), replacement=True)
            return DataLoader(ds, batch_size=batch_size, sampler=sampler, **kw)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kw)



except ImportError:          # numpy-only use still works
    def TorchBrainData(*a, **k):
        raise ImportError("torch is not installed")

    def make_loader(*a, **k):
        raise ImportError("torch is not installed")
