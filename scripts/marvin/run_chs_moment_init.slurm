#!/usr/bin/env bash
# CHS-only sweep with moment-based starting values, distributed across
# all 4 A100s on one sgpu_short node. Companion to the running 3-way
# sweep — different output cell (`*_chs_minit`) so it does not collide
# with the legacy-init CHS results.

#SBATCH --job-name=skillmodels-chs-minit
#SBATCH --account=ag_iame_gaudecker
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hmgaudecker@gmail.com
#SBATCH --output=logs/chs-minit_%j.out
#SBATCH --error=logs/chs-minit_%j.err

set -euo pipefail

SKILLMODELS_ROOT="${SKILLMODELS_ROOT:-$HOME/skillmodels-applications/skillmodels}"
SIM_REPRO_ROOT="${SIM_REPRO_ROOT:-$HOME/skillmodels-applications/sim_repro}"
export SIM_RESULTS_DIR="${SIM_RESULTS_DIR:-$HOME/sciebo_data/Skill estimation/Simulations/Results}"
export SIM_REPRO_OUT="${SIM_REPRO_OUT:-$SIM_REPRO_ROOT/estimates_3way_h2000}"

mkdir -p logs "$SIM_REPRO_OUT"

# User-installed pixi (AMD64-generic upstream) — not the SIGILL-prone
# `module load Pixi` build.
export PATH="$HOME/.pixi/bin:$PATH"
cd "$SKILLMODELS_ROOT"
echo "Using pixi: $(which pixi) $(pixi --version)"

nvidia-smi --list-gpus

VARIANT=translog
N=500

# 4 GPU workers, 125 sims each, GPU JAX (no JAX_PLATFORMS=cpu).
launch_chs_gpu() {
    local gpu_id="$1"
    local start="$2"
    local count="$3"
    CUDA_VISIBLE_DEVICES="$gpu_id" \
        pixi run -e tests-cuda13 python "$SIM_REPRO_ROOT/sim_sweep_chs_minit.py" \
        --variant "$VARIANT" \
        --n "$N" \
        --start "$start" \
        --count "$count" \
        > "logs/chs-minit_n${N}_gpu${gpu_id}_${SLURM_JOB_ID}.log" 2>&1 &
}

launch_chs_gpu 0   0 125
launch_chs_gpu 1 125 125
launch_chs_gpu 2 250 125
launch_chs_gpu 3 375 125

wait

echo
echo "All workers exited; counting per-cell coverage..."
pixi run -e tests-cuda13 python - <<'PY'
import os
import pickle
from pathlib import Path

root = Path(os.environ["SIM_REPRO_OUT"])
cell = root / "translog_n500_chs_minit"
if not cell.exists():
    print(f"{cell.name}: MISSING")
else:
    pkls = sorted(cell.glob("sim_*.pkl"))
    ok, fail = 0, 0
    for f in pkls:
        with f.open("rb") as fh:
            payload = pickle.load(fh)
        if payload.get("success"):
            ok += 1
        else:
            fail += 1
    print(f"{cell.name}: {ok} ok, {fail} failed (out of {len(pkls)})")
PY
