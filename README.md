# Diagnosing GNN Failure Modes with Layer-Wise Probe Entropy

Code and results for our LoG 2026 paper. Independent linear probes track
accuracy and normalized predictive entropy across frozen GNN layers.

## Setup

Tested with Python 3.13.4. Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

Plotting also requires LaTeX with `newtxtext`, `newtxmath`, `type1cm`,
`type1ec`, and `dvipng`.

## Reproduce the tables and figures

Run the following commands from the repository root to regenerate the
tables and figures from the saved results in `results/`:

```bash
python -m analysis.analyze_correlations
python -m analysis.analyze_alignment
python -m analysis.plot_figures
```

Statistics are written to `output/statistics/` and figures to
`output/figures/`. The paper PDFs are included in `figures/`.

| Paper result | Included inputs |
| --- | --- |
| Table 1 and Figure 3: four-diagnostic correlations | `results/correlations/` |
| Table 2: real-data entropy-confidence alignment | `results/probe_alignment/` |
| Table 4: synthetic entropy-confidence alignment | `results/synthetic_alignment/` |
| Figures 1, 2, 4, 5, and 6: layer and depth trajectories | `results/trajectories/` |

All four diagnostics use the same 120 four-layer runs across 12 datasets,
GCN and GAT, and five seeds. Correlations use the mean input-to-final-layer
changes in each of the 24 dataset-architecture settings.
Pearson and Spearman 95% confidence intervals
use 10,000 dataset-cluster-bootstrap draws, retaining paired GCN/GAT
settings, with bootstrap seed 20260826.

Real-data alignment reports the mean and sample standard deviation across
datasets after averaging five runs per dataset. Synthetic alignment reports
the mean and population standard deviation across five seeds.

Saved results include synthetic per-node arrays and real-data alignment
summaries.

## Run experiments

The scripts use seeds 42, 123, 456, 789, and 1024. GCN uses 32 hidden
features; GAT uses four heads with eight features each. Both use ReLU and
100 training epochs with Adam at learning rate 0.01. Use `--help` for
dataset, depth, and output options.

### Q1: Beneficial, neutral, and harmful smoothing

```bash
# Representative trajectories (Figure 1).
python -m experiments.q1.run_smoothing_trajectories --paired-splits

# Four-diagnostic comparison (Table 1, Figure 3, and log-ratio sensitivity).
python -m experiments.q1.run_diagnostic_comparison --paired-splits --save-checkpoints
python -m experiments.q1.evaluate_geometry --archive-dir output/controls
python -m analysis.analyze_correlations --run-dir output/checkpoint_geometry/runs --output output/new_correlations
```

### Q2: Failure-onset signatures across model depths

```bash
# Controlled depth sweeps (Figures 2, 4, and 5).
python -m experiments.q2.run_synthetic_gcn
python -m experiments.q2.run_synthetic_gat
python -m experiments.q2.run_synthetic_gat --graphs mid --depths 4 10 20

# Shallow and deep real-data trajectories (Figure 6).
python -m experiments.run_real --datasets cora roman-empire --depths 4 20 --skip-per-node --out output/real_trajectories
```

### Reliability: Node-level entropy-confidence alignment

The default real-data run covers all 12 datasets at depths 4 and 10.

```bash
# Real-data alignment (Table 2).
python -m experiments.run_real

# Synthetic alignment (Table 4).
python -m experiments.reliability.run_synthetic_alignment
```

`data/` contains fixed splits, filtered Chameleon/Squirrel graphs, and
GraphUniverse graphs. Other benchmark data are downloaded automatically.
Synthetic experiments use a fixed 60/20/20 train/validation/test split
with split seed 42. For datasets with multiple splits, runs and splits
are paired by index.

## Files

```text
log2026/
  src/             Models, linear probes, metrics, and data loaders
  experiments/
    run_real.py    Real-data trajectories and alignment
    q1/            Smoothing and diagnostic comparisons
    q2/            Failure-onset trajectories
    reliability/   Synthetic alignment
  analysis/        Statistics and figure generation
    plots/         Figure builders and shared styles
  data/            Fixed splits and synthetic graphs
  results/         Saved experimental measurements
  figures/         Paper figures: main/ and appendix/
  tests/           Metric and trajectory tests
```

Run the tests with `python -m unittest discover -s tests -v`.
