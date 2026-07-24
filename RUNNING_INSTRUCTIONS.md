# CSCN8020 Unitree G1 DQN — Running Instructions

These commands assume your project is in WSL at:

```text
/home/kevin/Unitree_MuJoCo_G1_Primer_Workshop
```

and your Python environment is:

```text
/home/kevin/.venvs/unitree/bin/python
```

## A. Copy the assignment files into your workshop repository

Extract the provided assignment package. Copy its contents into:

```text
~/Unitree_MuJoCo_G1_Primer_Workshop
```

Do not replace or edit:

```text
external/unitree_mujoco
```

The new `src/dqn` folder should sit beside the existing `src/g1_rl` folder.

Your structure should include:

```text
~/Unitree_MuJoCo_G1_Primer_Workshop/
    assets/
    external/
    src/
        g1_rl/
        dqn/
    requirements.txt
    requirements-assignment.txt
    run_full_assignment.sh
```

## B. Open WSL and activate the environment

From Windows PowerShell:

```powershell
wsl
```

Then in WSL:

```bash
cd ~/Unitree_MuJoCo_G1_Primer_Workshop
source ~/.venvs/unitree/bin/activate
```

Verify:

```bash
which python
python --version
```

The workshop repository officially targets Python 3.12. If your current environment is Python 3.14 and all required packages install and run, the code may work, but Python 3.12 is the safest match for the workshop setup.

## C. Install dependencies

First install the original workshop requirements:

```bash
python -m pip install -r requirements.txt
```

Then install assignment dependencies:

```bash
python -m pip install -r requirements-assignment.txt
```

Check PyTorch:

```bash
python -c "import torch; print(torch.__version__); print(torch.device('cpu'))"
```

Check MuJoCo:

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

## D. Make sure the Unitree external repository exists

Check:

```bash
ls external/unitree_mujoco
```

If it does not exist:

```bash
mkdir -p external
git clone https://github.com/unitreerobotics/unitree_mujoco.git external/unitree_mujoco
git -C external/unitree_mujoco checkout ae6a8403e272733e9996ef59990880330496177f
```

## E. Compile the code

```bash
PYTHONPATH=src python -m compileall src
```

## F. Validate the original environment

Headless:

```bash
PYTHONPATH=src python src/test_g1_elbow_env.py
```

Rendered:

```bash
PYTHONPATH=src python src/test_g1_elbow_env.py --render
```

The rendered command should show the robot moving its left elbow using the original rule-based validation policy.

## G. Run the DQN smoke test

```bash
PYTHONPATH=src python -m dqn.smoke_test
```

Expected ending:

```text
Smoke test PASSED
```

## H. Run the complete assignment automatically

This is the easiest option.

```bash
chmod +x run_full_assignment.sh
./run_full_assignment.sh
```

This runs:

- source compilation
- environment checker
- rule-based 20-episode baseline
- DQN smoke test
- Configuration A training (`0.995` decay)
- Configuration B training (`0.985` decay)
- greedy 20-episode evaluation for both
- automatic selection of the stronger DQN
- rule-based vs DQN comparison
- required plots
- report-ready measured results summary

Training is headless and CPU-compatible. Each configuration has a default maximum training time of 2.30 hours so the two training runs remain under the assignment's combined five-hour limit in normal use.

Do not close WSL while training.

## I. Run each step manually instead

Set the Python path for the current terminal:

```bash
export PYTHONPATH="$PYTHONPATH:src"
```

### 1. Rule-based baseline

```bash
python -m dqn.evaluate_rule_based \
  --output-dir results/rule_based \
  --seed 1000
```

### 2. Smoke test

```bash
python -m dqn.smoke_test
```

### 3. Train Configuration A

```bash
python -m dqn.train_dqn \
  --config-name config_a \
  --epsilon-decay 0.995 \
  --episodes 650 \
  --seed 42 \
  --device cpu \
  --max-hours 2.30
```

### 4. Train Configuration B

```bash
python -m dqn.train_dqn \
  --config-name config_b \
  --epsilon-decay 0.985 \
  --episodes 650 \
  --seed 42 \
  --device cpu \
  --max-hours 2.30
```

### 5. Evaluate Configuration A

Use `best.pt` if it exists:

```bash
python -m dqn.evaluate_dqn \
  --checkpoint models/config_a/best.pt \
  --output-dir results/config_a \
  --seed 1000 \
  --device cpu
```

If `best.pt` was not created, use:

```bash
python -m dqn.evaluate_dqn \
  --checkpoint models/config_a/final.pt \
  --output-dir results/config_a \
  --seed 1000 \
  --device cpu
```

### 6. Evaluate Configuration B

```bash
python -m dqn.evaluate_dqn \
  --checkpoint models/config_b/best.pt \
  --output-dir results/config_b \
  --seed 1000 \
  --device cpu
```

or use `models/config_b/final.pt` if necessary.

### 7. Select the stronger DQN

```bash
python -m dqn.compare_results
```

This creates:

```text
models/selected_dqn.pt
```

### 8. Compare selected DQN with rule-based policy

```bash
python -m dqn.compare_rule_based
```

### 9. Generate plots

```bash
python -m dqn.plot_results
```

### 10. Generate report values

```bash
python -m dqn.generate_report_summary
```

Open:

```text
report/Generated_Results_Summary.md
```

## J. Render the trained DQN robot

After training and selection:

```bash
PYTHONPATH=src python -m dqn.render_dqn_policy \
  --checkpoint models/selected_dqn.pt \
  --goals -0.8 -0.4 0.4 0.8 \
  --device cpu
```

This opens the MuJoCo viewer and runs the saved DQN greedily with epsilon `0.0`.

## K. Record the required 2–3 minute video

1. Start a Windows screen recorder.
2. Make sure both the MuJoCo viewer and WSL terminal are visible.
3. Run the render command above.
4. Capture at least two different goal angles; the default command shows four.
5. Show the terminal metrics and state whether each episode succeeds.
6. Stop recording after approximately 2–3 minutes.

The video must show the saved DQN checkpoint; do not use the rule-based policy as a substitute.

## L. Open the project in VS Code through WSL

From the project root:

```bash
code .
```

Select the WSL interpreter:

```text
/home/kevin/.venvs/unitree/bin/python
```

For notebook work, choose the WSL kernel you registered earlier. The DQN assignment scripts themselves should be run from the WSL terminal using the commands above.

## M. Common errors

### `No module named g1_rl` or `No module named dqn`

Run from the repository root and include:

```bash
export PYTHONPATH="$PYTHONPATH:src"
```

### `No module named torch`

```bash
python -m pip install torch
```

### MuJoCo viewer does not open

Training and evaluation do not need the viewer. For the video, confirm WSLg works by running:

```bash
python test_mujoco_viewer.py
```

### Training was interrupted

The current script saves `final.pt` when training exits normally. If WSL or the computer is closed abruptly, rerun that configuration. Do not combine partial metrics from different uncontrolled settings.

### Success rate below 80%

Keep the real result. First verify the code and environment are correct. Then, if time permits, increase `--episodes` while keeping both A/B experiments controlled and total training within five hours. Do not change the approved environment reward, observation, action set, or success condition.
