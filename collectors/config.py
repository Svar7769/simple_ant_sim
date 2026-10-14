from dataclasses import dataclass

@dataclass
class CollectorConfig:
    out_dir: str = "runs"
    run_name: str = ""

    collect_every: int = 1 # record ants
    snapshot_every: int = 10 # record grid
    save_grids: bool = True
