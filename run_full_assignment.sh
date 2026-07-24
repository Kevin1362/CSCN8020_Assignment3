#!/usr/bin/env bash
set -euo pipefail

# Run this script from the Unitree_MuJoCo_G1_Primer_Workshop repository root.
# The active Python environment must contain the workshop requirements,
# PyTorch, and matplotlib.

export PYTHONPATH="${PYTHONPATH:-}:src"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p results/config_a results/config_b results/rule_based results/comparison
mkdir -p models/config_a models/config_b report

echo "[1/11] Compile source"
$PYTHON_BIN -m compileall src

echo "[2/11] Validate supplied Gymnasium environment and original rule-based episode"
$PYTHON_BIN src/test_g1_elbow_env.py

echo "[3/11] Record 20-episode rule-based baseline"
$PYTHON_BIN -m dqn.evaluate_rule_based --output-dir results/rule_based --seed 1000

echo "[4/11] Run DQN smoke test"
$PYTHON_BIN -m dqn.smoke_test

echo "[5/11] Train Configuration A: epsilon decay 0.995"
$PYTHON_BIN -m dqn.train_dqn \
  --config-name config_a \
  --epsilon-decay 0.995 \
  --episodes 650 \
  --seed 42 \
  --device cpu \
  --max-hours 2.30

echo "[6/11] Train Configuration B: epsilon decay 0.985"
$PYTHON_BIN -m dqn.train_dqn \
  --config-name config_b \
  --epsilon-decay 0.985 \
  --episodes 650 \
  --seed 42 \
  --device cpu \
  --max-hours 2.30

A_CHECKPOINT="models/config_a/best.pt"
B_CHECKPOINT="models/config_b/best.pt"
[[ -f "$A_CHECKPOINT" ]] || A_CHECKPOINT="models/config_a/final.pt"
[[ -f "$B_CHECKPOINT" ]] || B_CHECKPOINT="models/config_b/final.pt"

echo "[7/11] Greedy evaluation of both DQN configurations (20 episodes each)"
$PYTHON_BIN -m dqn.evaluate_dqn \
  --checkpoint "$A_CHECKPOINT" \
  --output-dir results/config_a \
  --seed 1000 \
  --device cpu
$PYTHON_BIN -m dqn.evaluate_dqn \
  --checkpoint "$B_CHECKPOINT" \
  --output-dir results/config_b \
  --seed 1000 \
  --device cpu

echo "[8/11] Select stronger DQN and create models/selected_dqn.pt"
$PYTHON_BIN -m dqn.compare_results

echo "[9/11] Create rule-based vs selected-DQN comparison"
$PYTHON_BIN -m dqn.compare_rule_based

echo "[10/11] Generate required plots"
$PYTHON_BIN -m dqn.plot_results

echo "[11/11] Generate report-ready measured results summary"
$PYTHON_BIN -m dqn.generate_report_summary

echo
echo "FULL WORKFLOW COMPLETE"
echo "Selected checkpoint: models/selected_dqn.pt"
echo "Results: results/"
echo "Plots: results/*/plots and results/comparison/plots"
echo "Report values: report/Generated_Results_Summary.md"
echo "Next: run the rendered demonstration and screen-record 2-3 minutes."
