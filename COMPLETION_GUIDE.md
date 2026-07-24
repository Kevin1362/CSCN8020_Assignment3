# CSCN8020 Assignment 3 — Completion Guide

This guide explains **what you need to complete** for the Unitree G1 DQN assignment. It is separate from `RUNNING_INSTRUCTIONS.md`, which contains only the commands to run.

## 1. What the assignment requires

The assignment starts from the completed Unitree G1 Primer Workshop. You keep the supplied MuJoCo model, Gymnasium environment, PD controller, reward function, and success logic unchanged. Your work is the DQN layer: replay memory, Q-network, epsilon-greedy action selection, Bellman update, target network, training, evaluation, checkpointing, metrics, plots, comparison, and rendered demonstration.

The required DQN architecture is implemented in `src/dqn/q_network.py`:

- Input: 4 observation values
- Hidden layer 1: 64 units with ReLU
- Hidden layer 2: 64 units with ReLU
- Output: 3 raw Q-values

The three actions are:

- `0`: decrease controller target
- `1`: hold controller target
- `2`: increase controller target

The training goal is sampled from `[-0.8, +0.8]` radians. Final evaluation uses four benchmark goals: `-0.8`, `-0.4`, `+0.4`, and `+0.8`, five episodes each, for 20 episodes total.

## 2. Files included in this solution package

Copy the package contents into the root of your existing `Unitree_MuJoCo_G1_Primer_Workshop` repository.

The main student-written DQN files are:

```text
src/dqn/
    __init__.py
    common.py
    q_network.py
    replay_buffer.py
    agent.py
    smoke_test.py
    train_dqn.py
    evaluate_dqn.py
    evaluate_rule_based.py
    compare_results.py
    compare_rule_based.py
    plot_results.py
    render_dqn_policy.py
    generate_report_summary.py
```

Additional files:

```text
run_full_assignment.sh
requirements-assignment.txt
RUNNING_INSTRUCTIONS.md
COMPLETION_GUIDE.md
report/DQN_Assignment_Report_Template.md
```

## 3. Understand the code before submitting

You should be able to explain these points in your own words.

### Replay buffer

`ReplayBuffer` stores transitions from environment interaction. Each transition contains the current state, selected action, reward, next state, `terminated`, and `truncated`. The buffer has a maximum capacity of 50,000 transitions. Random mini-batches of 64 transitions are sampled for learning.

### Online and target Q-networks

The online network is updated by gradient descent. The target network is a separate copy used to calculate the Bellman target. It is synchronized with the online network every 250 optimization steps. This reduces rapid movement of the learning target and improves training stability.

### Epsilon-greedy exploration

During training, the agent selects a random action with probability epsilon and otherwise chooses the action with the highest predicted Q-value. Epsilon starts at 1.0 and cannot go below 0.05.

Two required experiments are used:

- Configuration A: epsilon decay = `0.995`
- Configuration B: epsilon decay = `0.985`

All other main hyperparameters and the seed are kept the same so the comparison is controlled.

### Bellman update

The implementation calculates:

```text
Q_target = reward + gamma * max(Q_target_network(next_state))
```

for non-terminal states. For a true `terminated` state, the bootstrap term is removed.

The code treats **time-limit truncation differently from true termination**. A truncated episode ends the rollout, but the transition is still allowed to bootstrap in the Bellman target because the underlying MDP state is not considered terminal. Explain this choice in the report.

### Loss

The implementation uses Huber loss (`smooth_l1_loss`) and gradient clipping for stability.

## 4. Complete the required workflow

Follow this order:

1. Validate the supplied Gymnasium environment.
2. Run and record the rule-based baseline.
3. Compile the source code.
4. Run the DQN smoke test.
5. Train Configuration A headlessly.
6. Train Configuration B headlessly.
7. Evaluate both DQN configurations using epsilon `0.0`.
8. Select the stronger configuration using success rate, stability, reward, and training time.
9. Compare the selected DQN with the rule-based policy.
10. Generate the plots.
11. Generate the measured results summary.
12. Render the saved DQN and record the 2–3 minute video.
13. Complete the technical report with your measured results and observations.
14. Verify the submission checklist.

## 5. Required evidence produced by the scripts

After a complete run, your repository should contain:

```text
results/
    config_a/
        training_metrics.csv
        loss_metrics.csv
        training_summary.json
        evaluation_episodes.csv
        evaluation_summary.csv
        evaluation_summary.json
        plots/
    config_b/
        ...same files...
    rule_based/
        evaluation_episodes.csv
        evaluation_summary.csv
        evaluation_summary.json
    comparison/
        epsilon_decay_comparison.csv
        selection.json
        rule_based_vs_dqn.csv
        rule_based_vs_dqn.json
        plots/

models/
    config_a/
        best.pt
        final.pt
    config_b/
        best.pt
        final.pt
    selected_dqn.pt

report/
    Generated_Results_Summary.md
```

## 6. Required plots

The plotting script generates:

- raw and moving-average training reward
- rolling training success rate
- epsilon over episodes
- training loss
- A/B greedy evaluation success comparison
- selected-DQN success rate by target angle

Use these images in the technical report.

## 7. How to choose the stronger DQN

The helper script uses this order:

1. higher 20-episode greedy evaluation success rate
2. higher final-50 training success rate
3. higher mean evaluation reward
4. shorter training time

This is only a helper. In your report, make an evidence-based recommendation using the plots and measured behavior as well. Do not argue only from one high reward value.

## 8. Final evaluation requirement

The selected DQN must be evaluated with epsilon exactly `0.0`.

The target is at least 16 successes in 20 episodes, equal to 80% success. If your result is lower, do not invent a higher result. Report the real result and discuss likely causes such as insufficient training, unstable exploration, oscillation, poor HOLD behavior, or inconsistent performance at the extreme targets.

## 9. Rule-based comparison discussion

Your report should answer:

- Which policy is more sample efficient?
- Which policy is more stable near the goal?
- Does DQN generalize across all four target angles?
- Does DQN learn to use HOLD appropriately?
- Is there oscillation or unnecessary target changing?
- Why can a hand-written policy outperform a learned policy on a simple control problem?

A strong explanation is that the rule-based controller starts with direct task knowledge and therefore needs no training samples, while DQN must discover useful state-action values through exploration. DQN is more general as a learning method but can be less efficient on a simple problem where a good control rule is already known.

## 10. Rendered video

Use `models/selected_dqn.pt`; do not retrain in the video.

The rendering command demonstrates four goals by default. Record the MuJoCo viewer and terminal console for approximately 2–3 minutes. The console prints success, episode length, final angle error, cumulative reward, and action counts.

During the video, state clearly that:

- the checkpoint is being loaded
- epsilon is `0.0`
- the displayed behavior is the learned DQN, not the rule-based policy
- the current target angle
- whether the episode succeeded

## 11. Technical report

Use `report/DQN_Assignment_Report_Template.md` as the starting point. After the experiments finish, open `report/Generated_Results_Summary.md` and transfer the measured values into the report.

Do not submit placeholder values.

## 12. Academic-integrity note

The assignment instructions state that AI-assisted code or writing must be disclosed according to course and institutional requirements. Because this package was AI-assisted, include the disclosure required by your course/instructor and make sure you personally understand every DQN component before submitting.
