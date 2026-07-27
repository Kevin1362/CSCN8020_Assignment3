# CSCN8020 Assignment 3 – Deep Q-Network Control of the Unitree G1 Robot

**Student Name:** Kevinkumar Patel  
**Student ID:** 8998612  
**Course:** CSCN8020 – Reinforcement Learning  

---

## Project Summary

This project implements a student-written Deep Q-Network (DQN) using PyTorch to control the left elbow joint of the Unitree G1 humanoid robot in the MuJoCo simulation environment.

The agent observes four values:

- Current elbow angle
- Elbow angular velocity
- Target elbow angle
- Difference between the current and target angles

Based on these observations, the DQN selects one of three discrete actions:

- Decrease the elbow target
- Hold the current position
- Increase the elbow target

The implementation includes:

- PyTorch Q-network
- Experience replay buffer
- Target network
- Epsilon-greedy exploration
- Bellman update
- Terminal-state masking
- Checkpoint saving and loading
- Training metrics and plots
- Greedy policy evaluation
- MuJoCo policy rendering

Two epsilon-decay configurations were trained under controlled conditions. Their performance was compared using reward, success rate, loss, exploration decay, episode length, and final error. The strongest model was saved as `models/selected_dqn.pt` and evaluated on four benchmark target angles.

---

## Final Validated Environment

The final validated run used:

- **Operating System:** Windows 11
- **Linux Environment:** WSL 2
- **Ubuntu Version:** Ubuntu 24.04 LTS
- **Python Version:** Python 3.12
- **Execution Device:** CPU
- **Simulator:** MuJoCo
- **Rendering Support:** WSLg

---

## GitHub Repository

**Repository URL:**

```text
https://github.com/Kevin1362/CSCN8020_Assignment3
```

**Cloneable Git URL:**

```text
https://github.com/Kevin1362/CSCN8020_Assignment3.git
```

Clone the repository:

```bash
git clone https://github.com/Kevin1362/CSCN8020_Assignment3.git
cd CSCN8020_Assignment3
```

---

## Quick Guide for the Evaluator

The most important assignment evidence is located in the following folders:

| Location | Purpose |
|---|---|
| `Unitree_MuJoCo_G1_Primer_Workshop.ipynb` | Completed notebook with environment checks, implementation, experiments, results, and interpretation |
| `src/dqn/` | Student-written DQN implementation |
| `models/selected_dqn.pt` | Final selected checkpoint |
| `results/config_a/` | Metrics and plots for epsilon-decay Configuration A |
| `results/config_b/` | Metrics and plots for epsilon-decay Configuration B |
| `results/comparison/` | Comparison outputs for both exploration configurations |
| `results/rule_based/` | Rule-based baseline results |
| `report/DQN_Assignment_Report.pdf` | Final technical report |
| `Video/` | Demonstration-video information or public video link |
| `README.md` | Setup, execution, output, and interpretation instructions |

For a quick verification:

1. Read this README.
2. Open the final report in `report/DQN_Assignment_Report.pdf`.
3. Review the metrics and plots in `results/`.
4. Confirm that `models/selected_dqn.pt` exists.
5. Run the evaluation command.
6. Run the rendering command.
7. Open the demonstration-video link in `Video/`.

---

## Create and Activate the Python Environment

### Option 1: Create a new environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Option 2: Activate the environment used for the validated run

```bash
source ~/.venvs/unitree/bin/activate
```

Verify the Python version:

```bash
python --version
```

Expected validated version:

```text
Python 3.12
```

---

## Install Dependencies

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

Add the project source folder to `PYTHONPATH`:

```bash
export PYTHONPATH="$PYTHONPATH:src"
```

The `PYTHONPATH` command should be run in each new terminal session before using the DQN modules.

---

## Clone the Unitree MuJoCo Dependency

The external Unitree MuJoCo dependency is not stored as a complete duplicate inside this repository.

Create the external folder:

```bash
mkdir -p external
```

Clone Unitree MuJoCo:

```bash
git clone https://github.com/unitreerobotics/unitree_mujoco.git \
external/unitree_mujoco
```

Checkout the validated commit:

```bash
git -C external/unitree_mujoco checkout \
ae6a8403e272733e9996ef59990880330496177f
```

---

## Run the Jupyter Notebook

Start Jupyter Lab:

```bash
jupyter lab Unitree_MuJoCo_G1_Primer_Workshop.ipynb
```

Open the notebook and run the cells in order.

The notebook contains:

- Environment validation
- Rule-based baseline verification
- DQN implementation
- Training workflow
- Epsilon-decay experiments
- Evaluation results
- Plot interpretation
- Policy comparison
- Final discussion

---

## Train the DQN

Run the complete assignment workflow:

```bash
chmod +x run_full_assignment.sh
./run_full_assignment.sh
```

The script performs the following tasks:

1. Trains epsilon-decay Configuration A.
2. Trains epsilon-decay Configuration B.
3. Saves the trained checkpoints.
4. Records training metrics.
5. Generates plots.
6. Evaluates both configurations.
7. Compares their performance.
8. Selects the final model.
9. Saves the selected model as:

```text
models/selected_dqn.pt
```

Training was completed using CPU execution and remained within the assignment's five-hour limit.

---

## Expected Outputs After Training

After the complete workflow finishes, outputs are organised into three main areas.

### 1. Trained Models

```text
models/
```

Important contents include:

| Output | Meaning |
|---|---|
| `models/config_a/` | Checkpoints produced during Configuration A |
| `models/config_b/` | Checkpoints produced during Configuration B |
| `models/selected_dqn.pt` | Final checkpoint selected for evaluation and rendering |

### 2. Metrics and Plots

```text
results/
```

Important subfolders include:

| Folder | Meaning |
|---|---|
| `results/config_a/` | Training data and plots for Configuration A |
| `results/config_b/` | Training data and plots for Configuration B |
| `results/comparison/` | Side-by-side comparison of both DQN configurations |
| `results/rule_based/` | Rule-based baseline results |
| `results/` CSV files | Structured evaluation, comparison, or summary data |

### 3. Final Documentation

```text
report/
```

The final technical report is:

```text
report/DQN_Assignment_Report.pdf
```

It explains the implementation, experimental method, results, comparison, limitations, and conclusions.

---

## Evaluate the Selected Checkpoint

Evaluation uses a deterministic greedy policy with:

```text
epsilon = 0.0
```

Run:

```bash
export PYTHONPATH="$PYTHONPATH:src"

python -m dqn.evaluate_dqn \
    --checkpoint models/selected_dqn.pt \
    --goals -0.8 -0.4 0.4 0.8 \
    --episodes-per-goal 5 \
    --device cpu
```

This command evaluates:

- Four target angles
- Five episodes per target
- Twenty total evaluation episodes

The target angles are:

```text
-0.8, -0.4, +0.4, and +0.8 radians
```

---

## Load and Verify the Saved Checkpoint

Use the following command to confirm that the selected checkpoint can be loaded:

```bash
export PYTHONPATH="$PYTHONPATH:src"

python -c "
import torch

checkpoint = torch.load(
    'models/selected_dqn.pt',
    map_location='cpu',
    weights_only=False
)

print('Checkpoint loaded successfully')
print('Checkpoint keys:', checkpoint.keys())
"
```

A successful load confirms that the selected model file is available and readable on CPU.

---

## Render the Selected Policy

Rendering requires a graphical MuJoCo environment such as WSLg.

Run:

```bash
export PYTHONPATH="$PYTHONPATH:src"

python -m dqn.render_dqn_policy \
    --checkpoint models/selected_dqn.pt \
    --goals -0.8 -0.4 0.4 0.8 \
    --device cpu \
    --pause-seconds 5
```

The renderer:

- Loads `models/selected_dqn.pt`
- Uses `epsilon = 0.0`
- Demonstrates all four target angles
- Displays the trained DQN controlling the robot
- Prints evaluation statistics in the terminal

---

## Understanding the Evaluation Output

During evaluation or rendering, the program reports values similar to:

```text
Target angle: -0.80
Success: True
Episode steps: 23
Final angle: -0.7942
Final absolute error: 0.0058
Cumulative reward: 10.9619
DECREASE actions: 13
HOLD actions: 7
INCREASE actions: 3
```

### Target Angle

The required elbow position for the episode, measured in radians.

### Success

Indicates whether the robot met the success condition for the target angle.

```text
True = successful episode
False = unsuccessful episode
```

### Episode Steps

The number of control decisions made before the episode ended.

Fewer steps normally indicate that the policy reached and stabilised near the target more quickly. However, success and final accuracy should also be considered.

### Final Angle

The elbow angle at the end of the episode.

It should be close to the assigned target angle.

### Final Absolute Error

The absolute difference between the final angle and the target angle:

```text
absolute error = |target angle - final angle|
```

A smaller value means that the policy finished closer to the target.

### Cumulative Reward

The total reward earned during the episode.

Higher reward generally represents:

- Accurate movement
- Faster convergence
- Stable behaviour
- Lower final error
- Successful completion

Reward should be interpreted together with success rate, episode length, and final error.

### Action Counts

The action counts show how often the DQN selected:

- `DECREASE`
- `HOLD`
- `INCREASE`

These values help explain the learned policy.

For example:

- More `DECREASE` actions are expected when moving toward a negative target.
- More `INCREASE` actions are expected when moving toward a positive target.
- `HOLD` actions near the goal indicate that the policy is attempting to stabilise the elbow instead of continually moving or oscillating.

---

## Understanding the Training Metrics

### Episode Reward

Episode reward is the total reward collected during one training episode.

An improving reward trend suggests that the agent is learning a more effective control policy.

Important interpretation:

- Higher reward is generally better.
- Individual episodes may fluctuate because exploration is active.
- The overall trend is more important than one isolated episode.

### Success Rate

Success rate represents the percentage of episodes in which the agent met the target condition.

It is calculated as:

```text
success rate = successful episodes / total episodes × 100
```

For final evaluation:

```text
20 successful episodes / 20 total episodes × 100 = 100%
```

### Training Loss

Loss measures the difference between:

- The Q-value predicted by the online network
- The Bellman target used for learning

Loss may fluctuate because the replay buffer contains experiences from different stages of training. A perfectly smooth or continuously decreasing loss is not required.

Loss should be interpreted with reward and success rate. A useful policy may still have variable training loss.

### Epsilon

Epsilon controls exploration.

- A high epsilon causes more random actions.
- A low epsilon causes more greedy actions.
- During training, epsilon decreases according to the selected decay schedule.
- During final evaluation and rendering, epsilon is set to `0.0`.

### Episode Length

Episode length is the number of steps required before success, termination, or truncation.

Shorter successful episodes usually indicate faster convergence, while longer episodes may indicate slower movement, extra corrections, or oscillation.

### Final Error

Final error measures how close the elbow is to the target at the end of the episode.

Smaller final error means better final positioning accuracy.

---

## Understanding the Generated Plots

The plots in `results/` provide visual evidence of learning and evaluation performance.

### Reward Plot

Shows the cumulative episode reward during training.

What to look for:

- Improving average reward
- Reduced instability near the end of training
- Similar or different learning behaviour between Configurations A and B

### Success-Rate Plot

Shows how frequently the agent successfully completed the control task.

What to look for:

- Increasing success during training
- Stable success near the end
- Whether the policy passes the required success threshold

### Epsilon-Decay Plot

Shows how exploration decreased over the training episodes.

What to look for:

- Starting epsilon
- Decay speed
- Minimum epsilon
- Difference between Configuration A and Configuration B

This plot supports the required exploration-decay comparison.

### Loss Plot

Shows the DQN optimisation loss.

What to look for:

- Whether updates remain finite
- Whether training becomes reasonably stable
- Whether there are signs of divergence

Loss should not be interpreted alone.

### Configuration Comparison Plot

Compares the performance of the two epsilon-decay experiments.

The final model was selected by considering:

- Success rate
- Mean evaluation reward
- Final error
- Episode length
- Stability
- Generalisation across all four targets

### Evaluation-by-Target Plot

Shows performance for:

- `-0.8` radians
- `-0.4` radians
- `+0.4` radians
- `+0.8` radians

This plot helps show whether the selected policy generalised to both positive and negative target positions.

### Rule-Based vs DQN Comparison

Compares the original rule-based controller with the selected DQN.

The comparison considers:

- Success count
- Mean reward
- Mean episode length
- Mean final error
- Stability
- Action behaviour

---
# DQN Demonstration Video

The video shows the trained DQN controlling the Unitree G1 robot using the selected checkpoint and greedy evaluation with `epsilon = 0.0`.

[Click here to watch the DQN demonstration video]((https://stuconestogacon-my.sharepoint.com/:v:/g/personal/kpatel8612_conestogac_on_ca/IQC4N7XfK695RZ9SIz2gEn9dATfM0R9nSIaJ8JpY_JP6Oi0?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=raY75s))


## Experimental Results

### Configuration A

- Training episodes: **650**
- Training time: **2.56 minutes**
- Final epsilon: **0.0500**
- Mean final-20 reward: **14.5631**
- Final-50 success rate: **100%**
- Evaluation success rate: **100%**
- Mean evaluation reward: **13.2136**

### Configuration B

- Training episodes: **650**
- Training time: **1.99 minutes**
- Final epsilon: **0.0500**
- Mean final-20 reward: **14.6910**
- Final-50 success rate: **100%**
- Evaluation success rate: **100%**
- Mean evaluation reward: **13.1641**

Both configurations reached a 100% evaluation success rate. Configuration A was selected because it produced a slightly higher mean evaluation reward.

---

## Final Evaluation Results

The selected DQN was tested for five episodes at each target.

| Target angle | Successful episodes | Mean reward |
|---:|---:|---:|
| `-0.8` radians | 5/5 | 10.9619 |
| `-0.4` radians | 5/5 | 15.6465 |
| `+0.4` radians | 5/5 | 15.4871 |
| `+0.8` radians | 5/5 | 10.7589 |

### Overall DQN Evaluation

- Total evaluation episodes: **20**
- Successful episodes: **20**
- Overall success rate: **100%**
- Mean evaluation reward: **13.2136**
- Evaluation epsilon: **0.0**

---

## Rule-Based and DQN Comparison

| Metric | Rule-based policy | Selected DQN |
|---|---:|---:|
| Successful episodes | 20/20 | 20/20 |
| Mean reward | 12.8666 | 13.2136 |
| Mean episode length | 24 steps | 21 steps |
| Mean final error | 0.01221 | 0.00397 |

Both policies completed all 20 evaluation episodes successfully.

The selected DQN achieved:

- Higher mean reward
- Shorter mean episode length
- Lower mean final error
- Accurate performance across all four target angles

These results indicate that the selected DQN learned an effective control policy and slightly improved the overall control quality compared with the rule-based baseline.

---

## Repository Structure

```text
CSCN8020_Assignment3/
│
├── README.md
├── requirements.txt
├── .gitignore
├── run_full_assignment.sh
├── Unitree_MuJoCo_G1_Primer_Workshop.ipynb
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
│   ├── config_a/
│   ├── config_b/
│   ├── comparison/
│   └── rule_based/
│
├── report/
│   └── DQN_Assignment_Report.pdf
│
├── Video/
│
└── external/
```

---

## Major Repository Files

| File or folder | Description |
|---|---|
| `Unitree_MuJoCo_G1_Primer_Workshop.ipynb` | Completed notebook containing environment validation, implementation, experiments, evaluation, and discussion |
| `src/dqn/` | Student-written Q-network, replay buffer, agent, training, evaluation, plotting, and rendering code |
| `src/g1_rl/` | Gymnasium environment and Unitree G1 robot-control support code |
| `models/config_a/` | Saved outputs for epsilon-decay Configuration A |
| `models/config_b/` | Saved outputs for epsilon-decay Configuration B |
| `models/selected_dqn.pt` | Final checkpoint selected for evaluation and rendering |
| `results/config_a/` | Configuration A metrics and plots |
| `results/config_b/` | Configuration B metrics and plots |
| `results/comparison/` | Exploration and performance comparisons |
| `results/rule_based/` | Rule-based baseline metrics |
| `report/DQN_Assignment_Report.pdf` | Final technical report |
| `Video/` | Demonstration-video instructions or public viewing link |
| `requirements.txt` | Python dependencies |
| `run_full_assignment.sh` | Complete training, evaluation, and output-generation workflow |
| `RUNNING_INSTRUCTIONS.md` | Additional setup and execution guidance |
| `COMPLETION_GUIDE.md` | Assignment completion information |
| `.gitignore` | Excludes environments, caches, builds, and temporary files |

---

## Student-Written DQN Implementation

The DQN implementation is located in:

```text
src/dqn/
```

It includes:

- A PyTorch Q-network with four observation inputs
- Three discrete action outputs
- ReLU hidden-layer activations
- Epsilon-greedy action selection
- Configurable epsilon decay
- Minimum epsilon
- Bounded replay-buffer storage
- Random mini-batch sampling
- Tensor conversion
- Bellman target calculation
- Terminal-state masking
- Online Q-network
- Target Q-network
- Target-network synchronisation
- Optimiser and loss calculation
- Checkpoint saving
- Checkpoint loading
- Reproducible random seeds
- CPU-compatible execution
- Greedy evaluation mode
- Training metrics
- Evaluation metrics
- Plot generation
- MuJoCo policy rendering

---

## Demonstration Video

The demonstration-video information is located in:

```text
Video/
```

The demonstration shows:

- The trained DQN checkpoint being loaded
- Greedy action selection with `epsilon = 0.0`
- MuJoCo rendering of the Unitree G1 robot
- At least two benchmark target angles
- Robot movement toward each target
- Terminal output showing success, reward, final error, and action counts

The public video link should be tested in a private or incognito browser window to confirm that the evaluator can open it without requesting access.

---

## How to Verify the Assignment

An evaluator can verify the project using the following process:

1. Clone the repository.
2. Create and activate the Python environment.
3. Install `requirements.txt`.
4. Clone the required Unitree MuJoCo dependency.
5. Export `PYTHONPATH`.
6. Open and review the notebook.
7. Review the rule-based baseline results.
8. Review Configuration A and Configuration B results.
9. Confirm that `models/selected_dqn.pt` exists.
10. Load the checkpoint using the provided command.
11. Run the 20-episode evaluation.
12. Confirm that evaluation uses `epsilon = 0.0`.
13. Review metrics and plots in `results/`.
14. Compare rule-based and DQN performance.
15. Run the rendering command.
16. Open the demonstration video.
17. Read the final technical report.

---

## Final Submission Summary

This repository contains:

- Completed Jupyter notebook
- Environment validation
- Rule-based baseline evidence
- Student-written DQN implementation
- Two epsilon-decay experiments
- CPU-compatible training
- Saved checkpoints
- Selected model checkpoint
- Twenty-episode greedy evaluation
- Overall success-rate calculation
- Rule-based and DQN comparison
- Metrics and plots
- Technical report
- Rendering code
- Demonstration-video information
- Setup and execution documentation

The final selected DQN achieved:

```text
20 successful episodes out of 20
100% overall evaluation success
13.2136 mean evaluation reward
```

---

## Repository URLs

**GitHub repository:**

```text
https://github.com/Kevin1362/CSCN8020_Assignment3
```

**Cloneable `.git` URL:**

```text
https://github.com/Kevin1362/CSCN8020_Assignment3.git
```

**Clone command:**

```bash
git clone https://github.com/Kevin1362/CSCN8020_Assignment3.git
```
