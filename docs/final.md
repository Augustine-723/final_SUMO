# NCHU main gate | Evaluation and submission

Authors: 林瑞裕、官奕愷. [English report](../submission/Report.pdf) · [Presentation](../submission/Presentation.pdf) · [Video](../submission/Demo.mp4) · [Code, checkpoints and raw results](../submission/Source_Code.zip).

The evaluation uses real site geometry and synthetic counts. Site-specific flow, signal timings and external calibration remain open. The Phase 6 [training run](https://github.com/Augustine-723/final_SUMO/actions/runs/45134925800) produced eight final checkpoints. The Phase 7 [evaluation run](https://github.com/Augustine-723/final_SUMO/actions/runs/45137746130) evaluated them with two baselines on 100 paired route files (SUMO 1.18). Each episode lasts 2,700 simulated seconds. The illustrative video was rendered locally with SUMO 1.24. Run versions and hashes accompany the models.

## Results

DQN reduced mean completed-vehicle wait relative to fixed timing in four profiles. The reduction ranged from 9.31% in balanced traffic to 26.65% off-peak. Under the gate reversal, the estimated DQN difference was less than 0.01 s. Queue-actuated control achieved the shortest mean vehicle wait among the core methods in every profile. PPO seed 22 had 152 incomplete trips in east-heavy demand. The pedestrian-reward pilot used training seed 11; the paired interval for pedestrian stopped time spans zero in the pedestrian peak. See [paired differences](../experiments/phase7/paired.csv), [means and standard deviations](../experiments/phase7/summary.csv) and [action diagnostics](../experiments/phase7/behavior.csv).

## Host and reproduction

In Antigravity, open the host checkout, inspect its origin and working tree, then fast-forward main. Preserve any uncommitted work and divergent history. From a clean main tracking this repository:

```bash
git remote get-url origin
git status --short --branch
git pull --ff-only origin main
(cd submission && sha256sum -c SHA256SUMS)
```

With no checkout, use `git clone https://github.com/Augustine-723/final_SUMO.git`. The three files named in the course proposal and the supplemental video are in `submission/`. Extract Source_Code.zip into a separate directory for the models and raw results. No retraining is required to inspect the outputs.

For a full rerun from the extracted archive with Python 3.11 and SUMO 1.18 (sumo/netconvert on PATH), install the CPU Torch build and requirements as in the Phase 7 workflow. Install reportlab and Pillow for document generation, FFmpeg and DejaVu / Noto Sans CJK fonts for rendering. From the extracted project root:

```bash
for profile in offpeak balanced directional reversal ped_peak; do
  for shard in 0 1 2 3; do
    python scripts/nchu_test_policies.py results/phase6 "$profile" "$shard" replay/shards
  done
done
python scripts/nchu_test_report.py replay/shards replay/analysis
python scripts/nchu_demo_video.py results/phase6 replay/video
python scripts/nchu_submission.py replay/analysis results/phase6 replay/video replay/submission
```

The report and slides describe the frozen experiment identified above. The CSV tables are rendered from results, while the discussion records the interpretation of that experiment. For a new experiment, update the interpretation and study identifiers before generating a new report.
