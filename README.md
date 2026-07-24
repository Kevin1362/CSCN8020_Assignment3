# CSCN8020 Assignment 3
## Deep Q-Network Control of the Unitree G1 Left Elbow

**Student Name:** Kevinkumar Patel  
**Student ID:** 8998612  
**Course:** CSCN8020 Reinforcement Learning  
**Instructor:** Prof. Enrique Espinosa  

## Project Summary

This project implements a student-written PyTorch Deep Q-Network to control
the Unitree G1 robot's left elbow in MuJoCo. The agent receives four observation
values and selects from three discrete actions: decrease, hold, or increase the
controller target. Two epsilon-decay settings were trained and compared. The
selected model was evaluated greedily on four benchmark target angles and was
also compared with the provided rule-based controller.

## Environment

- Operating system: Ubuntu through WSL on Windows
- Python version: Python 3.12.3
- Execution device: CPU
- Simulator: MuJoCo
- RL interface: Gymnasium
- Framework: PyTorch

## Create and Activate Environment

```bash
python3 -m venv ~/.venvs/unitree
source ~/.venvs/unitree/bin/activate