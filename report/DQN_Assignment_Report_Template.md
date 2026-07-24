# Deep Q-Network Control of the Unitree G1 Left Elbow

**Course:** CSCN8020 — Reinforcement Learning  
**Assignment:** DQN Assignment  
**Student:** Kevinkumar Patel  
**Instructor:** Prof. Enrique Espinosa  
**Date:** 24/07/2026

---

## 1. Introduction

This project extends the Unitree MuJoCo G1 Primer Workshop by replacing the existing hand-written high-level elbow decision policy with a learned Deep Q-Network (DQN). The supplied workshop already provides the fixed-base Unitree G1 model, a Gymnasium-compatible environment, low-level proportional-derivative control, bias-force compensation, success logic, and a deterministic rule-based validation policy. Therefore, the main focus of this assignment is the reinforcement-learning layer rather than redesigning the robot model or low-level controller.

The objective is to train one DQN agent that can move the Unitree G1 left elbow toward goals sampled from a limited multi-goal range and keep the elbow inside the required success tolerance for enough consecutive environment steps. The learned policy uses three discrete actions: decrease the internal controller target, hold the target, or increase it. The final trained policy is evaluated greedily at four required benchmark angles: -0.8 rad, -0.4 rad, +0.4 rad, and +0.8 rad.

DQN is suitable for this task because the action space is small and discrete. The observation is continuous, so a table-based Q-learning method would be impractical without discretization. A neural network can instead approximate the action-value function and estimate one Q-value for each of the three possible actions from the four-value observation vector.

## 2. Environment Definition

The environment used in this project is `G1ElbowTargetEnv`. The controlled joint is `left_elbow_joint`, and the corresponding actuator is `left_elbow`. The robot is fixed at the base, which keeps the experiment focused on elbow target control rather than balance or locomotion.

### 2.1 Observation Space

The observation has four continuous values:

1. current elbow angle
2. current elbow angular velocity
3. goal angle
4. goal angle minus current elbow angle

The fourth value gives the current signed angle error directly. A negative error indicates that the current elbow angle is above the goal, while a positive error indicates that it is below the goal.

### 2.2 Action Space

The action space contains three discrete actions:

- Action 0: decrease the internal controller target
- Action 1: hold the internal controller target
- Action 2: increase the internal controller target

The DQN does not directly output motor torque. Instead, it chooses how the internal target should change. The supplied PD controller then converts the target into actuator torque while MuJoCo bias-force compensation helps account for gravity and other model forces. This separation allows the DQN to learn high-level decisions while the low-level controller handles continuous torque control.

### 2.3 Reward and Success

The environment reward is mainly based on the absolute elbow angle error. Smaller error gives a better reward. The environment also gives a bonus when the elbow enters the success region and discourages unnecessary non-HOLD actions when the arm is already close to the goal. A larger terminal bonus is added when the success condition is achieved.

An episode is considered successful only when the elbow stays inside the success tolerance for the required number of consecutive environment steps. The default maximum episode length is 150 steps. The assignment evaluation uses four fixed benchmark goals with five episodes per goal, giving 20 final evaluation episodes.

### 2.4 Termination and Truncation

The implementation handles `terminated` and `truncated` separately. A true `terminated` transition represents successful completion of the task, so the Bellman target does not bootstrap from the next state. A `truncated` transition occurs because the environment reached the time limit. The rollout still stops, but the DQN implementation allows Bellman bootstrapping for that transition because the underlying state is not treated as a true terminal MDP state. This distinction prevents the time limit itself from being interpreted as an absorbing terminal condition.

## 3. DQN Architecture

The Q-network was implemented in PyTorch using the required baseline architecture:

- Input layer: 4 observation values
- Hidden layer 1: 64 units with ReLU activation
- Hidden layer 2: 64 units with ReLU activation
- Output layer: 3 Q-values

No softmax is applied to the output layer because DQN needs unconstrained action-value estimates rather than action probabilities. The selected action during greedy evaluation is the action with the maximum Q-value.

The implementation creates two neural networks. The online network is updated by gradient descent, while the target network is used to calculate the bootstrap term of the Bellman target. The target network is initialized from the online network and synchronized every 250 optimization steps.

## 4. Replay Buffer and Transition Handling

Experience replay is implemented using a bounded replay buffer with a capacity of 50,000 transitions. Each stored transition contains:

- state
- action
- reward
- next state
- terminated flag
- truncated flag

Training does not begin immediately. The agent first collects at least 500 transitions, which satisfies the required warm-up period. After warm-up, random mini-batches of 64 transitions are sampled from the replay buffer. Random sampling reduces the strong temporal correlation that would exist if the network learned only from consecutive transitions.

## 5. Bellman Target and Optimization

For every sampled transition, the online network predicts Q-values for the current state. The Q-value corresponding to the action that was actually taken is selected. The target network then estimates the maximum Q-value of the next state.

For non-terminal transitions, the target is:

**target = reward + gamma × max next-state Q-value**

The discount factor is 0.95. For true terminated transitions, the bootstrap term is removed, so the target becomes only the received reward.

The implementation uses Huber loss between the selected current Q-values and Bellman targets. Huber loss is less sensitive to large errors than mean-squared error and can improve training stability. The Adam optimizer is used with a learning rate of 0.001. Gradient clipping is also applied to reduce the risk of unstable parameter updates.

## 6. Exploration Strategy

The training policy uses epsilon-greedy exploration. At the beginning of training, epsilon is 1.0, meaning that actions are initially selected randomly. After each episode, epsilon is multiplied by the selected decay factor but is not allowed to fall below 0.05.

Two controlled configurations were tested:

- Configuration A: epsilon decay = 0.995
- Configuration B: epsilon decay = 0.985

Configuration A keeps exploration active for a longer period. Configuration B shifts toward exploitation earlier. All other required baseline hyperparameters and the random-seed policy were kept the same so that epsilon decay was the main controlled difference between the two experiments.

During final evaluation, epsilon is set to 0.0, so the policy is fully greedy and no random exploration is used.

## 7. Training Methodology and Reproducibility

Training was performed headlessly without opening the MuJoCo viewer. This improves speed and keeps the experiment within the assignment's five-hour training limit. The implementation supports CPU execution and does not require CUDA.

Random seeds were set for Python's `random` module, NumPy, PyTorch, the Gymnasium environment resets, and the environment action space. Both exploration-decay experiments used the same main seed policy. Training goals were sampled from the approved range of -0.8 to +0.8 radians.

The required baseline hyperparameters were:

| Hyperparameter | Value |
|---|---:|
| Discount factor | 0.95 |
| Learning rate | 0.001 |
| Mini-batch size | 64 |
| Replay capacity | 50,000 |
| Initial epsilon | 1.00 |
| Minimum epsilon | 0.05 |
| Target update interval | 250 optimization steps |
| Warm-up | 500 transitions |
| Maximum episode length | 150 steps |
| Training goal range | [-0.8, +0.8] rad |
| Evaluation epsilon | 0.00 |

The implementation records episode reward, success, episode length, final absolute error, epsilon, loss, goal angle, termination status, truncation status, and elapsed time. Checkpoints are saved for each experiment, and the stronger configuration is copied to `models/selected_dqn.pt` for final demonstration.

## 8. Exploration-Decay Experiment Results

| Metric | Configuration A (0.995) | Configuration B (0.985) |
|---|---:|---:|
| Total training episodes | 650 | 650 |
| Wall-clock training time | 2.56 min | 1.99 min |
| Final epsilon | 0.0500 | 0.0500 |
| Mean cumulative reward, final 20 episodes | 14.5631 | 14.6910 |
| Training success rate, final 50 episodes | 100.0% | 100.0% |
| Greedy evaluation success rate, 20 episodes | 100.0% | 100.0% |
| Mean evaluation reward | 13.2136 | 13.1641 |

Both configurations completed 650 episodes and achieved 100% success over the final 50 training episodes as well as 100% success during the 20-episode greedy evaluation. Configuration B trained slightly faster and achieved a slightly higher mean reward over the final 20 training episodes. Configuration A, however, achieved the higher mean evaluation reward and was therefore selected as the final DQN according to the predefined selection rule.

### 8.1 Training Reward

The raw reward for Configuration A was very noisy at the beginning of training because the agent was exploring heavily and had not yet learned a reliable policy. Several early episodes had strongly negative rewards. However, the 20-episode moving average improved quickly and became positive as learning progressed. After the early exploration phase, the reward trend became much more stable and stayed around the mid-teen range.

The final-20 mean reward was 14.5631 for Configuration A and 14.6910 for Configuration B. This shows that both configurations reached strong late-stage training performance. Configuration B had a slightly higher late-training reward, but Configuration A produced the stronger mean evaluation reward after exploration was completely disabled.

### 8.2 Training Success Rate

The rolling success-rate plots show that both agents improved quickly and eventually reached a stable success rate of 100%. Configuration A rose from roughly the mid-50% range and reached a rolling success rate of 1.00 at around episode 120. After that point, it remained at 100% for the remainder of training.

Configuration B reached a rolling success rate of 1.00 earlier, at approximately episode 90 to 100, and also remained there for the rest of training. This suggests that the faster epsilon decay allowed Configuration B to move toward exploitation earlier. However, both configurations were equally successful by the end of training.

### 8.3 Epsilon Decay

The epsilon-decay plots clearly show the difference between the two exploration schedules. Configuration A used the slower decay value of 0.995, so exploration continued for much longer. Its epsilon value gradually approached the minimum value of 0.05 near the later part of training.

Configuration B used the faster decay value of 0.985. Its epsilon fell much more quickly and reached the minimum value of 0.05 at approximately episode 200. This means Configuration B began exploiting its learned policy earlier, while Configuration A continued collecting a wider variety of exploratory experiences.

In this experiment, both schedules were effective. Configuration B became consistently successful earlier and trained faster, while Configuration A produced the slightly higher final mean evaluation reward.

### 8.4 Loss

The Huber-loss plots for both configurations were not perfectly smooth and did not steadily decrease throughout training. This is expected in reinforcement learning because the target distribution changes as the policy and replay buffer change.

Configuration A showed a gradual increase in average loss with fluctuations but remained numerically stable. Configuration B showed a similar pattern with slightly lower peak values. Neither training run showed evidence of exploding loss or numerical divergence. Therefore, the optimization process remained stable for both configurations even though the loss curves were noisy and changed over time.

## 9. Selected Configuration and Final Evaluation

The selected configuration was **Configuration A (epsilon decay = 0.995)**. Both configurations achieved identical 100% greedy evaluation success and identical 100% success over the final 50 training episodes. The selection was therefore decided by the next comparison criterion, mean evaluation reward. Configuration A achieved a mean evaluation reward of 13.2136 compared with 13.1641 for Configuration B.

The final greedy evaluation used epsilon = 0.0 and included five episodes for each required benchmark target.

| Goal | Episodes | Successes | Success rate | Mean reward |
|---|---:|---:|---:|---:|
| -0.8 rad | 5 | 5 | 100.0% | 10.9619 |
| -0.4 rad | 5 | 5 | 100.0% | 15.6465 |
| +0.4 rad | 5 | 5 | 100.0% | 15.4871 |
| +0.8 rad | 5 | 5 | 100.0% | 10.7589 |
| Overall | 20 | 20 | 100.0% | 13.2136 |

The required target was at least 80% success, equivalent to at least 16 successful episodes out of 20. The measured result was **20/20**, which corresponds to **100.0%**. Therefore, the selected DQN exceeded the required performance threshold.

The DQN achieved 100% success at every target angle, showing strong generalization across negative and positive directions and across both smaller and larger target magnitudes. Based on mean reward, the easiest target was -0.4 rad with a mean reward of 15.6465. The hardest target was +0.8 rad with a mean reward of 10.7589. The larger-magnitude targets produced lower mean rewards because they required the elbow to travel farther from its initial position and therefore required more control adjustments.

## 10. Rule-Based Baseline Compared with DQN

The supplied rule-based policy moves the internal controller target directly toward the final goal. When the controller target is sufficiently close, it selects HOLD and allows the low-level PD controller to settle the elbow. The selected DQN was evaluated on the same four target angles and the same total number of benchmark episodes.

| Metric | Rule-based policy | Selected DQN |
|---|---:|---:|
| Successes / 20 | 20 | 20 |
| Success rate | 100.0% | 100.0% |
| Mean cumulative reward | 12.8666 | 13.2136 |
| Mean episode length | 24.00 | 21.00 |
| Mean final absolute error | 0.01221 | 0.00397 |
| Main qualitative behaviour | Direct task-informed movement toward the target with deterministic control and reliable settling | Learned target-seeking behaviour with equally reliable success, shorter episodes, and better final precision |

### 10.1 Sample Efficiency

The rule-based policy is more sample efficient in the strict sense because it does not need to learn from experience. Its behavior is based on direct knowledge of the relationship between the controller target and the goal. The DQN begins without this task-specific decision rule and must collect transitions through interaction before useful Q-values can be learned. Therefore, even though the DQN eventually achieved strong performance, it required many more environment samples than the hand-written baseline.

### 10.2 Stability Near the Goal

The selected DQN showed strong stability near the goal. Its mean final absolute error was 0.00397 rad compared with 0.01221 rad for the rule-based policy. The DQN also completed episodes in an average of 21 steps compared with 24 steps for the rule-based controller. Its mean cumulative reward was also slightly higher.

These results suggest that the DQN reached the target more precisely and settled somewhat faster than the rule-based controller. The low final error and shorter episode length are consistent with stable control near the desired angle.

### 10.3 Generalization Across Goals

The DQN generalized successfully across all four required benchmark goals. It achieved 5 out of 5 successful episodes at -0.8 rad, -0.4 rad, +0.4 rad, and +0.8 rad. This confirms that the policy did not only learn one direction or one target magnitude. Instead, it learned a high-level control strategy that worked across the complete evaluation range.

Although the larger-magnitude targets had lower mean rewards, they still achieved 100% success. Therefore, the difference in reward reflects greater control effort rather than a failure to generalize.

### 10.4 HOLD Action and Oscillation

The evaluation summary did not include detailed action-count data, so the exact frequency of the HOLD action cannot be calculated from the metrics alone. However, the selected DQN achieved 100% evaluation success, a short mean episode length, and a very small final absolute error. These results do not show evidence of severe oscillation or repeated failure to settle near the target.

During the rendered demonstration, the console should be used to observe the DECREASE, HOLD, and INCREASE action counts. If HOLD appears near the target and the elbow settles with limited movement, this confirms that the DQN learned an appropriate stopping behavior. If a few increase/decrease corrections occur near the end of an episode, they can be described as minor corrective actions rather than major instability.

### 10.5 Why the Rule-Based Policy May Outperform DQN

A hand-written controller can outperform a learned DQN on a simple task when the correct decision logic is already easy to specify. The rule-based policy has an explicit understanding of the goal direction and controller target, so it can act correctly from the first episode. DQN must approximate the same useful behavior from sampled rewards and transitions. Its performance can be affected by exploration, limited training, approximation error, and the distribution of sampled goals.

The advantage of DQN is not necessarily better performance on this simple problem. Instead, the assignment demonstrates how a useful control policy can be learned from experience without directly coding the final high-level decision rule. In this experiment, both policies achieved 100% success, while the DQN also achieved slightly better mean reward, shorter episodes, and lower final error.

## 11. Discussion of Failures, Stability, and Generalization

No complete failures were observed during the final 20-episode greedy evaluation because the selected DQN achieved 20 successes out of 20 episodes. Most instability occurred during the early training stage, when reward values were noisy and success rates were still increasing. This behavior was expected because the agent was still exploring and the replay buffer was collecting a wide variety of experiences.

As training progressed, the moving-average reward became positive and stable, and the rolling success rate reached 100%. Once this point was reached, the policy remained consistently successful for the remainder of training. The final evaluation results confirm that the learned behavior remained reliable after exploration was completely disabled.

The larger target magnitudes, -0.8 rad and +0.8 rad, produced lower mean rewards than the smaller-magnitude targets. This suggests that longer elbow movements required more control effort. However, the policy still completed every episode successfully, so this did not represent a generalization failure.

The available metrics also do not show evidence of severe oscillation near the goal. The DQN achieved a mean final error of only 0.00397 rad and required fewer steps than the rule-based controller. The rendered demonstration should be used to confirm the exact HOLD behavior and whether the elbow makes any minor corrective movements before settling.

The exploration-decay schedule influenced how quickly each model moved from exploration to exploitation. Configuration B reached stable 100% rolling success earlier because its epsilon decayed faster. Configuration A explored for longer but achieved the slightly higher mean evaluation reward. This suggests that the additional exploration in Configuration A may have improved the final greedy policy slightly, although the difference between the two configurations was small.

## 12. Evidence-Based Recommendation

Based on the measured results, **Configuration A (epsilon decay = 0.995)** is recommended as the stronger exploration-decay setting.

The recommendation is based on the following evidence:

- 100.0% greedy evaluation success
- 100.0% training success over the final 50 episodes
- mean evaluation reward of 13.2136
- successful performance at all four target angles
- stable reward and success trends after the early training period

Configuration B trained faster and had a slightly higher mean reward over the final 20 training episodes. However, both configurations achieved the same perfect success rates, while Configuration A achieved the slightly better mean evaluation reward. Since final greedy performance is more important than one late-training reward metric, Configuration A was selected as the final model.

The difference between the two configurations was relatively small, so both epsilon-decay settings were effective for this task. Configuration B provided faster convergence, while Configuration A provided slightly stronger final greedy evaluation performance.

## 13. Limitations and Future Improvements

One limitation is that training is restricted to a relatively small one-joint control problem with a fixed-base robot. The learned policy does not address full-body balance, locomotion, or coordinated multi-joint control. Another limitation is that the experiment compares only two epsilon-decay values while holding the other required baseline hyperparameters constant.

The experiment also used one main random-seed policy for the controlled comparison. Future work could repeat the complete training process with multiple seeds to measure performance variance and determine whether Configuration A remains consistently stronger.

Future improvements could include experimenting with Double DQN to reduce Q-value overestimation, prioritized replay to focus learning on more informative transitions, or different network sizes. Additional analysis could record detailed action counts during evaluation so that HOLD usage and possible oscillation can be measured directly. Any change to the approved reward function or success condition would require instructor approval.

## 14. Conclusion

This assignment implemented a student-written PyTorch DQN for discrete high-level control of the Unitree G1 left elbow. The agent mapped a four-value continuous observation to three action-value estimates and learned from replayed transitions using an online network, target network, Bellman updates, epsilon-greedy exploration, and Huber-loss optimization.

Two exploration-decay settings were compared under controlled conditions. Both configurations achieved 100% success over the final 50 training episodes and 100% success in the final greedy evaluation. Configuration A, using epsilon decay 0.995, was selected because it achieved the slightly higher mean evaluation reward.

The selected DQN achieved 20 successful episodes out of 20, exceeding the required 80% performance threshold. It generalized successfully across all four benchmark target angles. Compared with the rule-based baseline, both policies achieved 100% success, but the DQN achieved a higher mean cumulative reward, shorter mean episode length, and lower mean final absolute error.

Overall, the experiment demonstrates that a DQN can learn an effective multi-goal control policy from interaction with the environment. At the same time, the comparison shows that a rule-based controller remains more sample efficient when strong task knowledge is already available. The final rendered demonstration loads the saved DQN checkpoint and uses epsilon = 0.0, confirming that the demonstrated behavior comes from the learned policy rather than the original rule-based controller.

---

## Appendix A — Main Execution Commands

```bash
cd ~/Unitree_MuJoCo_G1_Primer_Workshop
source ~/.venvs/unitree/bin/activate
export PYTHONPATH="$PYTHONPATH:src"

python -m dqn.evaluate_rule_based --output-dir results/rule_based --seed 1000
python -m dqn.smoke_test

python -m dqn.train_dqn --config-name config_a --epsilon-decay 0.995 --episodes 650 --seed 42 --device cpu --max-hours 2.30
python -m dqn.train_dqn --config-name config_b --epsilon-decay 0.985 --episodes 650 --seed 42 --device cpu --max-hours 2.30

python -m dqn.evaluate_dqn --checkpoint models/config_a/best.pt --output-dir results/config_a --seed 1000 --device cpu
python -m dqn.evaluate_dqn --checkpoint models/config_b/best.pt --output-dir results/config_b --seed 1000 --device cpu

python -m dqn.compare_results
python -m dqn.compare_rule_based
python -m dqn.plot_results
python -m dqn.generate_report_summary

python -m dqn.render_dqn_policy --checkpoint models/selected_dqn.pt --goals -0.8 -0.4 0.4 0.8 --device cpu
```

## Appendix B — AI-Assistance Disclosure

AI tools were used to help organize, edit, and improve the clarity and formatting of the written report. The code was executed in my own environment, and the training, evaluation, checkpoints, numerical results, and plots reported in this assignment were generated from my own experiment runs. I reviewed the final submission and verified the reported values before submission.