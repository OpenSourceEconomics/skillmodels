#!/usr/bin/env bash
# SLURM batch script for the translog three-way comparison on Marvin.
#
# Compares (at n_halton=2000, panel n=500, 500 sims each):
#   1. AF Stage A only         (initialization_strategy="moment_based",
#                               two_stage_measurement=False) — sigma_meas
#                               in the AF MLE chain.
#   2. AF Stage A + Stage B    (two_stage_measurement=True) — Spearman
#                               pre-step pins sigma_meas; eliminates the
#                               sigma_inv ridge.
#   3. CHS                     (Kalman-filter MLE on the CHS-flavoured
#                               spec; runs on CPU).
#
# Layout: AF gets all 4 A100s (2 per AF variant, 250 sims each). CHS
# runs on the node's CPUs in parallel (8 workers × ~63 sims each).
#
# Layout assumption:
#   $HOME/skillmodels-applications/                    # parent workspace
#   $HOME/skillmodels-applications/skillmodels/        # this repo (af-estimator branch)
#   $HOME/skillmodels-applications/sim_repro/          # sim runner code
#   $HOME/sciebo_data/Skill estimation/Simulations/    # MATLAB results data
#
# Submit with:
#   sbatch scripts/marvin/run_three_way_translog_n2k.slurm

#SBATCH --job-name=skillmodels-translog-3way-n2k
#SBATCH --account=ag_iame_gaudecker
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hmgaudecker@gmail.com
#SBATCH --output=logs/translog-3way-n2k_%j.out
#SBATCH --error=logs/translog-3way-n2k_%j.err

set -euo pipefail

# ---------------------------------------------------------------
# Environment
# ---------------------------------------------------------------
SKILLMODELS_ROOT="${SKILLMODELS_ROOT:-$HOME/skillmodels-applications/skillmodels}"
SIM_REPRO_ROOT="${SIM_REPRO_ROOT:-$HOME/skillmodels-applications/sim_repro}"
export SIM_RESULTS_DIR="${SIM_RESULTS_DIR:-$HOME/sciebo_data/Skill estimation/Simulations/Results}"
export SIM_REPRO_OUT="${SIM_REPRO_OUT:-$SIM_REPRO_ROOT/estimates}"

mkdir -p logs "$SIM_REPRO_OUT"

module load Pixi
cd "$SKILLMODELS_ROOT"

nvidia-smi --list-gpus

N_HALTON=2000
N_SIMS=500
VARIANT=translog
N=500

# ---------------------------------------------------------------
# AF workers: 4 GPUs split 2-vs-2 across stage-A / stage-A+B.
# 250 sims per GPU, 500 sims total per variant.
# ---------------------------------------------------------------
launch_af_worker() {
    local gpu_id="$1"
    local stage_flag="$2"   # --no-two-stage-measurement | --two-stage-measurement
    local stage_tag="$3"    # stagea | stageab
    local start="$4"
    local count="$5"
    CUDA_VISIBLE_DEVICES="$gpu_id" \
        pixi run -e tests-cuda12 python "$SIM_REPRO_ROOT/sim_sweep.py" \
        --variant "$VARIANT" \
        --n "$N" \
        --start "$start" \
        --count "$count" \
        --n-halton "$N_HALTON" \
        --out-suffix "_h${N_HALTON}" \
        $stage_flag \
        > "logs/af_${stage_tag}_n${N}_gpu${gpu_id}_${SLURM_JOB_ID}.log" 2>&1 &
}

# Stage A: GPUs 0 and 1, 250 sims each.
launch_af_worker 0 "--no-two-stage-measurement" "stagea" 0   250
launch_af_worker 1 "--no-two-stage-measurement" "stagea" 250 250

# Stage A+B: GPUs 2 and 3, 250 sims each.
launch_af_worker 2 "--two-stage-measurement"    "stageab" 0   250
launch_af_worker 3 "--two-stage-measurement"    "stageab" 250 250

# ---------------------------------------------------------------
# CHS workers: 8 CPU-only workers, ~63 sims each.
#
# Forcing JAX to CPU keeps GPU memory free for the AF workers and
# avoids contention. The 16 cpus-per-task on the node accommodate
# 8 CHS workers comfortably; CHS is single-threaded per process for
# the optimizer step.
# ---------------------------------------------------------------
launch_chs_worker() {
    local idx="$1"
    local start="$2"
    local count="$3"
    JAX_PLATFORMS=cpu \
        pixi run -e tests-cpu python "$SIM_REPRO_ROOT/sim_sweep_chs.py" \
        --variant "$VARIANT" \
        --n "$N" \
        --start "$start" \
        --count "$count" \
        > "logs/chs_n${N}_w${idx}_${SLURM_JOB_ID}.log" 2>&1 &
}

# Split 500 sims into 8 contiguous chunks of 63 (last chunk gets
# 65 to cover 500 = 7*63 + 65 = 446 + 54... let me redo).
# Actually: 500 / 8 = 62.5. Use chunks of 63 with the last absorbing
# the remainder via the `min(start+count, n_total)` clamp in
# sim_sweep_chs.py.
launch_chs_worker 0   0 63
launch_chs_worker 1  63 63
launch_chs_worker 2 126 63
launch_chs_worker 3 189 63
launch_chs_worker 4 252 63
launch_chs_worker 5 315 63
launch_chs_worker 6 378 63
launch_chs_worker 7 441 63   # absorbs sims 441..499

wait

echo
echo "All workers exited; computing per-cell coverage..."
pixi run -e tests-cpu python - <<'PY'
import os
import pickle
from pathlib import Path

root = Path(os.environ["SIM_REPRO_OUT"])
cells = (
    "translog_n500_stagea_h2000",
    "translog_n500_stageab_h2000",
    "translog_n500_chs",
)
for cell in cells:
    if not (root / cell).exists():
        print(f"{cell}: MISSING")
        continue
    pkls = sorted((root / cell).glob("sim_*.pkl"))
    ok, fail = 0, 0
    for f in pkls:
        with f.open("rb") as fh:
            payload = pickle.load(fh)
        if payload.get("success"):
            ok += 1
        else:
            fail += 1
    print(f"{cell}: {ok} ok, {fail} failed (out of {len(pkls)})")
PY
