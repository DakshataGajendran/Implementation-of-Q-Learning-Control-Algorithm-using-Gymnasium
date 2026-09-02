# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement
The objective is to train an autonomous agent to navigate a slippery 4x4 grid world (`FrozenLake-v1`) from the starting position (S) to the goal state (G) while avoiding hazardous holes (H). Because the surface is slippery, the agent's movement direction is uncertain and only partially dependent on the chosen action. Using the model-free **Q-Learning** algorithm, the agent must iteratively discover an optimal policy that maximizes cumulative discounted rewards despite the stochastic environment dynamics.

## Software Requirements
1. Python: 3.8+
2. Gymnasium: gymnasium>=0.29.0
3. NumPy: numpy>=1.22.0
4. Matplotlib: matplotlib>=3.5.0
5. Jupyter Notebook / Google Colab
---

## Environment Description
The `FrozenLake-v1` environment represents a 4x4 grid of frozen and broken ice tiles:

* **Grid Layout**:
```text
S F F F
F H F H
F F F H
H F F G
```

* **Legend**:
  * `S`: Starting point (safe)
  * `F`: Frozen surface (safe)
  * `H`: Hole (fall into hole -> episode terminates with 0 reward)
  * `G`: Goal (reach goal -> episode terminates with +1 reward)

* **State Space**: Discrete space of 16 states (0 to 15).

* **Action Space**: Discrete space with 4 possible actions:
  * `0`: Left
  * `1`: Down
  * `2`: Right
  * `3`: Up

* **Transition Dynamics**: With `is_slippery=True`, taking an action moves the agent in the intended direction with probability 1/3, and in either of the two perpendicular directions with probability 1/3 each.

* **Reward Mechanism**:
  * Reaching Goal (`G`): +1
  * Falling into Hole (`H`): 0
  * Moving to Frozen tile (`F`): 0


## Theory
