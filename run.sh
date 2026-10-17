#!/usr/bin/env bash
# Generate the training dataset: many seeds, random food layouts, varied colony size.
# Usage:  bash run.sh            (defaults below)
#         N_SEEDS=5 STEPS=300 bash run_all.sh   (quick test)
set -euo pipefail

cd "$(dirname "$0")"

N_SEEDS=${N_SEEDS:-20}          # seeds 0 .. N_SEEDS-1
STEPS=${STEPS:-1500}
OUT_DIR=${OUT_DIR:-runs/dataset_v0}
JOBS=${JOBS:-4}                 # parallel runs

PATCHES=(2 3 4)                 # number of food patches, cycled over seeds
ANTS=(50 100 150)               # colony size, cycled over seeds

mkdir -p "$OUT_DIR"

run_one () {
    local seed=$1
    local p=${PATCHES[$(( seed % ${#PATCHES[@]} ))]}
    local a=${ANTS[$(( (seed / ${#PATCHES[@]}) % ${#ANTS[@]} ))]}
    local name=$(printf "seed%03d_p%d_a%d" "$seed" "$p" "$a")

    if [[ -f "$OUT_DIR/$name/traj.npz" ]]; then
        echo "skip $name (exists)"
        return
    fi
    echo "run  $name"
    if ! MPLBACKEND=Agg python run.py --headless --random_food \
        --seed "$seed" --steps "$STEPS" \
        --n_patches "$p" --n_ants "$a" \
        --out_dir "$OUT_DIR" --name "$name" \
        > "$OUT_DIR/$name.log" 2>&1; then
        echo "FAIL $name (see $OUT_DIR/$name.log)"
    fi
}

for seed in $(seq 0 $(( N_SEEDS - 1 ))); do
    run_one "$seed" &
    while (( $(jobs -rp | wc -l) >= JOBS )); do sleep 0.5; done
done
wait

n_ok=$(ls "$OUT_DIR"/seed*/traj.npz 2>/dev/null | wc -l)
echo "done: $n_ok / $N_SEEDS runs saved in $OUT_DIR"
(( n_ok == N_SEEDS )) || { echo "some runs failed, check the .log files"; exit 1; }
