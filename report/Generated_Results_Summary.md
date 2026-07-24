# Generated Results Summary — Completed Interpretation

> This version uses the measured results from the completed training and evaluation runs. The interpretation of HOLD behavior is written cautiously because the summary file does not contain action-count data from the rendered demo.

## Exploration-Decay Comparison

| Metric | Configuration A (0.995) | Configuration B (0.985) |
|---|---:|---:|
| Training episodes | 650 | 650 |
| Wall-clock training time (min) | 2.56 | 1.99 |
| Final epsilon | 0.0500 | 0.0500 |
| Mean reward, final 20 training episodes | 14.5631 | 14.6910 |
| Training success rate, final 50 | 100.0% | 100.0% |
| Final greedy evaluation success | 100.0% | 100.0% |
| Mean evaluation reward | 13.2136 | 13.1641 |

**Selected configuration:** Configuration A (`config_a`, epsilon decay = 0.995)

Configuration A and Configuration B both showed very strong and stable learning by the end of training. Both achieved a 100% training success rate over the final 50 episodes and a 100% greedy evaluation success rate. Configuration B had a slightly higher mean reward over the final 20 training episodes and completed training faster. However, Configuration A achieved the slightly higher mean evaluation reward, 13.2136 compared with 13.1641 for Configuration B. Because the evaluation success rates and final training success rates were tied, Configuration A was selected using the defined selection rule, which next considered mean evaluation reward.

## Selected DQN Final Evaluation

| Goal | Episodes | Successes | Success rate | Mean reward |
|---|---:|---:|---:|---:|
| -0.8 rad | 5 | 5 | 100.0% | 10.9619 |
| -0.4 rad | 5 | 5 | 100.0% | 15.6465 |
| +0.4 rad | 5 | 5 | 100.0% | 15.4871 |
| +0.8 rad | 5 | 5 | 100.0% | 10.7589 |
| Overall | 20 | 20 | 100.0% | 13.2136 |

The selected DQN achieved 20 successful episodes out of 20, giving an overall success rate of 100%. Therefore, the agent clearly met the required 80% success threshold of at least 16 successful episodes. The evaluation was completed with epsilon set to 0.0, so the agent used a fully greedy learned policy. The results also show that the DQN generalized successfully across all four required benchmark targets, including both positive and negative elbow angles.

Based on mean reward, the easiest target was -0.4 rad, where the DQN achieved 5 out of 5 successful episodes with the highest mean reward of 15.6465. The hardest target was +0.8 rad, which still achieved 5 out of 5 successes but had the lowest mean reward of 10.7589. The larger target magnitudes, -0.8 rad and +0.8 rad, produced lower mean rewards than the smaller target magnitudes. This may be because the elbow had to move farther from its initial position, which required more control actions and created more opportunity for overshoot or correction.

## Rule-Based Baseline vs Selected DQN

| Metric | Rule-based | Selected DQN |
|---|---:|---:|
| Successes / 20 | 20 | 20 |
| Success rate | 100.0% | 100.0% |
| Mean cumulative reward | 12.8666 | 13.2136 |
| Mean episode length | 24.00 | 21.00 |
| Mean final absolute error | 0.01221 | 0.00397 |

### Sample Efficiency

The rule-based policy is more sample efficient because it begins with task-specific knowledge. It already knows whether the controller target should move toward the goal and does not need to learn through trial-and-error interactions. The DQN, on the other hand, starts without this high-level decision rule and must collect transitions, explore different actions, store experiences in the replay buffer, and gradually learn useful Q-values. Therefore, the rule-based policy is more sample efficient even though the final DQN achieved slightly better quantitative performance.

### Stability Near the Goal

The selected DQN showed strong stability near the goal based on the measured evaluation results. It achieved a mean final absolute error of 0.00397 rad, which was lower than the rule-based policy's 0.01221 rad. The DQN also completed episodes in an average of 21 steps compared with 24 steps for the rule-based policy, and its mean cumulative reward was slightly higher at 13.2136 compared with 12.8666.

These results suggest that the selected DQN reached the target more precisely and, on average, completed the task more efficiently than the rule-based policy. The quantitative results do not show evidence of major instability near the goal because the DQN achieved 100% success and maintained a very small final error.

### HOLD Action and Oscillation

The generated summary does not include detailed action-count data, so the exact frequency of the HOLD action cannot be confirmed from the metrics alone. However, the 100% success rate, shorter mean episode length, and very low final absolute error suggest that the learned policy was able to settle near the target without severe or repeated oscillation.

During the rendered demonstration, the most important behavior to confirm is whether the DQN selects HOLD when the elbow is already close to the target or whether it continues switching between INCREASE and DECREASE. If the rendered demo shows limited movement near the final angle, this supports the conclusion that the DQN learned to use HOLD appropriately. If repeated action switching is visible, that should be mentioned as a minor limitation even though the quantitative performance remained strong.

### Generalization Across Goals

The DQN generalized successfully across all four benchmark target angles. It achieved a 100% success rate at -0.8 rad, -0.4 rad, +0.4 rad, and +0.8 rad. This is important because it shows that the policy did not only learn one direction or one target magnitude. Instead, it learned a control strategy that worked across both positive and negative goals and across both smaller and larger target magnitudes.

The lower mean rewards at ±0.8 rad suggest that the larger target movements were somewhat more demanding, but the agent still completed every evaluation episode successfully.

## Discussion of Failures, Stability, and Generalization

No complete evaluation failures were observed because the selected DQN achieved 20 successes in 20 episodes. Both epsilon-decay configurations also achieved 100% success over their final 50 training episodes and 100% greedy evaluation success. This indicates that the learning process became highly reliable by the end of training.

The main difference between the two exploration schedules was relatively small. Configuration B, with the faster epsilon decay of 0.985, achieved a slightly higher mean reward over the final 20 training episodes and finished training more quickly. Configuration A, with the slower epsilon decay of 0.995, achieved a slightly higher mean evaluation reward. Since both configurations reached the same success rates, the evidence suggests that both exploration schedules were effective for this task.

Configuration A was selected because the comparison rule prioritized greedy evaluation success, final training success, and then mean evaluation reward. The first two metrics were tied, so the slightly higher evaluation reward of Configuration A became the deciding factor.

The final evaluation results also show good generalization. The DQN succeeded at every required target angle. The smallest final error and shorter episode length compared with the rule-based policy suggest that the learned policy was stable and precise near the goal. The only behavior that still needs to be judged visually is whether the agent used HOLD smoothly or made unnecessary action changes close to the target.

## Evidence-Based Recommendation

Configuration A, with epsilon decay 0.995, is recommended as the stronger exploration-decay setting for the final submitted DQN. Both configurations achieved 100% success in the final 50 training episodes and 100% success during greedy evaluation. Configuration B trained faster and had a slightly higher mean reward in the final 20 training episodes, but Configuration A achieved the higher mean evaluation reward of 13.2136 compared with 13.1641.

Because the main goal is to select a policy that performs consistently when exploration is disabled, the slightly stronger greedy evaluation reward supports the selection of Configuration A. The difference between the two settings is small, so the result also suggests that both epsilon-decay schedules were suitable for this relatively simple one-joint control task.

## Limitations and Future Improvements

One limitation of this experiment is that it focuses on a single elbow joint of a fixed-base Unitree G1 robot. The learned policy does not address full-body balance, locomotion, or coordinated control of multiple joints. Another limitation is that only two epsilon-decay settings were compared while the other baseline hyperparameters were kept constant.

The experiment also used one main random-seed policy for the controlled comparison. Future work could repeat training with several different random seeds to measure variability and determine whether the selected configuration remains consistently better. Other possible improvements include testing Double DQN to reduce Q-value overestimation, prioritized experience replay, or different network sizes. Additional analysis could also record action counts during evaluation to measure how often the DQN uses HOLD and to identify any unnecessary oscillation near the goal.

Any change to the approved reward function, success condition, or environment design would need instructor approval for this assignment.

## Conclusion

This assignment successfully implemented and evaluated a student-written PyTorch DQN for high-level control of the Unitree G1 left elbow. The DQN mapped the four-value observation vector to three Q-values representing DECREASE, HOLD, and INCREASE actions. The agent learned through replay memory, an online Q-network, a target Q-network, Bellman updates, epsilon-greedy exploration, and Huber-loss optimization.

Two epsilon-decay configurations were compared under the same training conditions. Both achieved 100% training success over the final 50 episodes and 100% greedy evaluation success. Configuration A, using epsilon decay 0.995, was selected because it achieved the slightly higher mean evaluation reward.

The selected DQN achieved 20 successful episodes out of 20, exceeding the required 80% success threshold. It generalized successfully across all four benchmark goals. Compared with the rule-based policy, both policies achieved 100% success, but the DQN produced a higher mean cumulative reward, a shorter mean episode length, and a lower mean final absolute error.

Overall, the experiment shows that DQN can learn an effective and accurate multi-goal elbow-control policy from experience. At the same time, the rule-based policy remains more sample efficient because it begins with task-specific knowledge and does not require training. The final results demonstrate both the strengths of learned control and the practical value of comparing reinforcement-learning policies with simple transparent baselines.
