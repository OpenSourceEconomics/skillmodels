# Snellius runner: translog AF simulation sweep

Batch scripts for re-running the AF sweep on the Snellius `gpu_h100` partition (4 ×
NVIDIA H100 SXM5, 64 cores, 768 GiB RAM per node).

## What runs

`run_translog_sim.slurm` launches the translog sim sweep across all four H100 GPUs on a
single node, using **two estimators** in parallel:

- **AF** (Antweiler-Freyberger): the period-by-period MLE with Halton quadrature. Each
  GPU sweeps a disjoint slice of the 500 stored simulations (125 sims/GPU).
- **CHS** (Cunha-Heckman-Schennach via UKF Kalman filter): same datasets, same
  measurement-system normalisations (first loading=1 + all intercepts pinned to 0), but
  investment is treated as a regular latent factor (CHS lacks AF's `is_endogenous`
  notion). Each GPU also runs a CHS slice for the corresponding 125 sims.

The two estimators write to disjoint output directories (`translog_n500/` for AF,
`translog_n500_chs/` for CHS) so a downstream aggregator can diff their parameter
recovery.

H100 vs local RTX 3070: per-sim AF wall-clock drops from ~8 min to roughly 60–90 s, so
500 sims complete in 30–45 min instead of ~3 days. CHS is much cheaper per-sim
(seconds), so the CHS sweep finishes well before AF.

## One-time Snellius setup

On a login node (compute nodes have no internet):

```bash
# Clone repo
cd $HOME
git clone <skillmodels-applications-url> skillmodels-applications
cd skillmodels-applications/skillmodels

# Install pixi if not already
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc

# Install the tests-cuda12 environment (~10 min, downloads jax+CUDA)
pixi install -e tests-cuda12

# Copy the MATLAB simulation result files from your local sciebo.
# Replace USER and SOURCE with your local Snellius transfer endpoint:
mkdir -p $HOME/sciebo_data/Skill\ estimation/Simulations
rsync -av USER@local:'~/sciebo/Skill\ estimation/Simulations/Results/' \
    "$HOME/sciebo_data/Skill estimation/Simulations/Results/"

# Make the sim_repro/ directory available (it lives next to skillmodels/
# in the workspace; if not in your clone, copy it across):
ls $HOME/skillmodels-applications/sim_repro/sim_sweep.py
```

## Submitting the job

```bash
cd $HOME/skillmodels-applications/skillmodels
sbatch scripts/snellius/run_translog_sim.slurm
```

The script writes per-GPU logs to `logs/sweep_translog_n*_gpu*_<jobid>.log` and per-sim
pickles to `$SIM_REPRO_ROOT/estimates/translog_n{500,2000}/`. A short success/failure
summary is printed at the end.

## Tunables (env vars)

- `SKILLMODELS_ROOT`: where this repo lives (default:
  `$HOME/skillmodels-applications/skillmodels`)
- `SIM_REPRO_ROOT`: where the sim runner code lives (default:
  `$HOME/skillmodels-applications/sim_repro`)
- `SIM_RESULTS_DIR`: where the MATLAB `.mat` result files live (default:
  `$HOME/sciebo_data/Skill estimation/Simulations/Results`)
- `SIM_REPRO_OUT`: where output pickles are written (default:
  `$SIM_REPRO_ROOT/estimates`)

## Pulling results back

After the job finishes:

```bash
rsync -av USER@snellius:'~/skillmodels-applications/sim_repro/estimates/translog_n500/' \
    /home/hmg/econ/skillmodels-applications/sim_repro/estimates/translog_n500/
```

Then run the local aggregator/report writer over the merged pickles.

## Notes on the sweep itself

- The Halton count is 10000 per axis (matches MATLAB). H100's 94 GiB HBM2e can fit much
  higher Halton counts, so feel free to bump `--n-halton 20000` for sharper integration
  if you want — per-sim time goes up roughly linearly with Halton.
- The truth-based `start_params` warm start in `sim_sweep.py` keeps the optimiser away
  from the `phi` upper bound (committed in `aea7b86`). With the corrected
  `log_ces_with_constant` spec (committed in `281ff84`), translog sims recover the
  production parameters within ~5% relative bias on local hardware.
- See `obsidian/.../simulation-replication-status-2026-05-03.md` for background on the
  sweep design.
