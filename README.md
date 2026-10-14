# ML-Based-Noise-Source-Fingerprinting-for-Quantum-Circuits

Can the **type** and rough **strength** of quantum noise be recovered from ordinary computational-basis
measurement counts? Five small entangled-state circuits (Bell, GHZ, W, 4-qubit GHZ, two Bell pairs) are
simulated with Qiskit Aer under four noise channels (depolarizing, amplitude damping, phase damping,
readout error) at six strengths. Six physics-motivated features plus raw outcome probabilities feed five
classical classifiers (Random Forest, XGBoost, SVM, MLP, logistic regression). A systematic ablation varies
shots, dataset size, severity range, circuit family, feature set, hyperparameters, tuning and CV folds.
All results are from **simulation**; no quantum hardware was used and no quantum advantage is claimed.

## Key results (2,048 shots, pooled model)
- Random Forest: **80.2 +/- 1.1 %** accuracy for the noise type (chance = 25 %); up to 95.3 % for a per-circuit model (4-qubit GHZ).
- Shots dominate: 43.9 % at 50 shots -> 94.0 % at 32,768 shots. Tree-ensemble hyperparameters change accuracy by at most about 2 points once depth >= 8.
- Accuracy tracks the number of shots that fall outside the ideal outcome set (10 % with none, 99.7 % with more than 1,000).
- Transfer to an unseen circuit is weak, and raw probabilities transfer at chance unless the ideal supports overlap.

## Repository layout
```
code/       common.py (circuits, noise, features, data generation)   expcore.py (models, CV)
            runexp.py (ablations)   analysis.py (diagnostics)   agg.py / figs.py (aggregation, figures)
            gen_all.py (regenerates datasets, deterministic seeds)
datasets/   36 seeded datasets: s<shots>_d<draw>.csv  (12 shot settings x 3 draws, 720 rows each)
results/    raw per-evaluation results (*.jsonl) and extra.json, analysis.json
figs/       figures (pdf/png) used in the paper
```
Paths are resolved relative to the repository folder, so the code runs from wherever it is cloned.

## Quick start
```bash
pip install -r requirements.txt
python code/gen_all.py 600        # (optional) regenerate datasets; the number is a time budget in seconds
python code/runexp.py shots 600   # run an ablation: shots | size | feat | sev | circ | hp | cv | main | tune
python code/analysis.py           # diagnostics -> results/analysis.json
python code/figs.py               # redraw all figures into figs/
```
Run the commands from the repository root. Results are cached in `results/`, so finished runs are skipped.

## Reproducibility notes
- Library versions (Qiskit, Aer, scikit-learn, xgboost) were not recorded when the original results were produced, so `requirements.txt` is unpinned. Small numerical differences are possible with other versions.
- Random stratified cross-validation shares circuit/channel/severity configurations between training and test folds; it measures recognition of a configuration from a fresh sample, not generalisation to unseen severities.
