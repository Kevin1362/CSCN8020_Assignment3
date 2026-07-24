# CSCN8020 Assignment 3 – Deep Q-Network Control of the Unitree G1 Robot

**Student Name:** Kevinkumar Patel  
**Student ID:** 8998612  
**Course:** CSCN8020 – Reinforcement Learning

---

# Project Summary

This project implements a student-developed Deep Q-Network (DQN) using PyTorch to control the left elbow joint of the Unitree G1 humanoid robot in the MuJoCo simulator.

The implementation includes:

- Deep Q-Network with PyTorch
- Experience Replay Buffer
- Target Network
- Epsilon-Greedy Exploration
- Bellman Update
- Checkpoint Saving and Loading
- Training Metrics
- Performance Evaluation
- Policy Rendering in MuJoCo

Two exploration-decay configurations were trained and compared. The best-performing model was selected and evaluated on four different target elbow positions using a deterministic (greedy) policy.

---

# Operating Environment

The final validated implementation was executed using:

- Operating System: Windows 11
- Linux Environment: WSL2 (Ubuntu 24.04 LTS)
- Python Version: 3.12
- Device: CPU
- Simulator: MuJoCo
- Visualization: WSLg

---

# GitHub Repository

Repository:

```text
https://github.com/Kevin1362/CSCN8020_Assignment3
```

Clone URL:

```bash
git clone https://github.com/Kevin1362/CSCN8020_Assignment3.git
```

---

# Create / Activate Python Environment

Create a new environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Or activate the validated environment:

```bash
source ~/.venvs/unitree/bin/activate
```

Verify Python:

```bash
python --version
```

---

# Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install project dependencies:

```bash
python -m pip install -r requirements.txt
```

Export the project source path:

```bash
export PYTHONPATH="$PYTHONPATH:src"
```

---

# Clone the Unitree MuJoCo Dependency

```bash
mkdir -p external

git clone https://github.com/unitreerobotics/unitree_mujoco.git \
external/unitree_mujoco

git -C external/unitree_mujoco checkout ae6a8403e272733e9996ef59990880330496177f
```

---

# Run the Jupyter Notebook

```bash
jupyter lab Unitree_MuJoCo_G1_Primer_Workshop.ipynb
```

Run all notebook cells in order.

---

# Train the DQN

Run the complete assignment workflow:

```bash
chmod +x run_full_assignment.sh
./run_full_assignment.sh
```

This script:

- trains Configuration A
- trains Configuration B
- saves checkpoints
- generates metrics
- generates plots
- performs final evaluation
- saves the selected checkpoint

---

# Evaluate the Selected Checkpoint

```bash
export PYTHONPATH="$PYTHONPATH:src"

python -m dqn.evaluate_dqn \
    --checkpoint models/selected_dqn.pt \
    --goals -0.8 -0.4 0.4 0.8 \
    --episodes-per-goal 5 \
    --device cpu
```

---

# Load the Saved Checkpoint

```bash
export PYTHONPATH="$PYTHONPATH:src"

python -c "
import torch
checkpoint=torch.load(
'models/selected_dqn.pt',
map_location='cpu',
weights_only=False
)
print(checkpoint.keys())
"
```

---

# Render the Selected Policy

```bash
export PYTHONPATH="$PYTHONPATH:src"

python -m dqn.render_dqn_policy \
    --checkpoint models/selected_dqn.pt \
    --goals -0.8 -0.4 0.4 0.8 \
    --device cpu \
    --pause-seconds 5
```

The rendering uses a greedy policy (epsilon = 0).

---

# Repository Structure

```
CSCN8020_Assignment3/
│
├── Unitree_MuJoCo_G1_Primer_Workshop.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── run_full_assignment.sh
│
├── src/
│   ├── dqn/
│   └── g1_rl/
│
├── models/
│   ├── config_a/
│   ├── config_b/
│   └── selected_dqn.pt
│
├── results/
│
├── report/
│
├── external/
│
└── Video/
```

---

# Major Repository Files

| File | Description |
|------|-------------|
| Unitree_MuJoCo_G1_Primer_Workshop.ipynb | Complete notebook containing environment validation, implementation, experiments, evaluation, and discussion. |
| src/dqn/ | Student-written Deep Q-Network implementation. |
| src/g1_rl/ | Robot environment and simulator interface. |
| models/selected_dqn.pt | Final selected trained model checkpoint. |
| models/config_a/ | Saved checkpoints for Configuration A. |
| models/config_b/ | Saved checkpoints for Configuration B. |
| results/ | Training metrics, plots, evaluation results, and comparison data. |
| report/ | Assignment report and generated result summary. |
| requirements.txt | Python dependencies. |
| run_full_assignment.sh | Complete training and evaluation workflow. |
| RUNNING_INSTRUCTIONS.md | Additional execution instructions. |
| COMPLETION_GUIDE.md | Assignment completion guide. |
| .gitignore | Files excluded from Git tracking. |

---

# Student-Written DQN Implementation

The DQN implementation was written using PyTorch and Gymnasium.

The implementation includes:

- Neural network with four state inputs and three action outputs
- Experience Replay Buffer
- Target Network
- Bellman Update
- Epsilon-Greedy Exploration
- Target Network Synchronization
- Checkpoint Saving
- Checkpoint Loading
- CPU-compatible execution
- Deterministic evaluation mode
- Performance plotting
- MuJoCo rendering

---

# Final Evaluation Summary

The selected DQN policy was evaluated on four target joint angles:

- -0.8 radians
- -0.4 radians
- +0.4 radians
- +0.8 radians

Five evaluation episodes were executed for each target.

Final Results:

- Evaluation Success Rate: **20 / 20**
- Overall Success Rate: **100%**
- Mean Evaluation Reward: **13.2136**

---

# Repository URLs

GitHub Repository

```
https://github.com/Kevin1362/CSCN8020_Assignment3
```

Clone URL

```bash
git clone https://github.com/Kevin1362/CSCN8020_Assignment3.git
```

---

# Notes

This repository contains the complete submission for CSCN8020 Assignment 3, including the notebook, DQN implementation, trained models, evaluation scripts, rendering scripts, generated results, and documentation required for evaluation.