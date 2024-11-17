
# Dirichlet Normal Gamma Monte Carlo Tree Search (DNG-MCTS)

This repository contains an implementation of the **Dirichlet Normal Gamma Monte Carlo Tree Search (DNG-MCTS)** algorithm, as described in the paper: [DNG-MCTS: Monte Carlo Tree Search with Uncertainty Estimates](https://proceedings.neurips.cc/paper/2013/hash/846c260d715e5b854ffad5f70a516c88-Abstract.html). The DNG-MCTS algorithm is a variant of MCTS that leverages the Dirichlet Normal Gamma (DNG) distribution to model uncertainty in the search process, making it effective for solving complex decision-making problems in stochastic environments.

## 📋 Features

* **DNG-MCTS Algorithm** : An enhanced MCTS algorithm using Dirichlet Normal Gamma distribution to manage exploration and exploitation effectively.
* **Custom Environment Support** : Includes an implementation for the `RaceTrack` environment.
* **Checkpointing** : Save and load MCTS checkpoints for evaluation and analysis.
* **Visualization** : Supports human-readable rendering of the environment during evaluation.
* **Modular Design** : Easily extendable to other environments and MCTS-based algorithms.

## 📂 Project Structure

.

├── base_mcts.py            # Base MCTS implementation

├── dng_mcts.py             # DNG-MCTS algorithm implementation
├── node.py                 # Node definition for MCTS
├── race_track_env/         # Custom RaceTrack environment
├── main.py                 # Main script for training and evaluation
├── README.md               # Project README file
└── checkpoints/            # Directory for saving model checkpoints


## 🚀 Getting Started

### Prerequisites

Ensure you have the following Python packages installed:

* Python 3.8+
* `numpy`
* `pickle` (standard library)
* `argparse` (standard library)


### Clone the Repository

`git clone https://github.com/Copticoder/dng-mcts.git
 cd dng-mcts`


### Running the DNG-MCTS Algorithm

To train the DNG-MCTS algorithm on the `RaceTrack` environment, run:

```bash
python main.py --algorithm dng --env_name a --train --max_episodes 500000
```

To evaluate the trained model using checkpoints, run:

```bash
python main.py --algorithm dng --env_name a --max_episodes 500000 --eval_episode_number 5000
```

Arguments

* `--algorithm`: Algorithm to use (`dng` by default).
* `--env_name`: Name of the environment (`a` by default).
* `--env_dynamics`: Enable stochastic environment dynamics.
* `--train`: Run in training mode.
* `--max_episodes`: Maximum number of episodes for training.
* `--eval_episode_number`: Number of episodes to evaluate during testing.

## 🛠️ Code Overview

### `DNG_MCTS` Class

The `DNG_MCTS` class extends `MCTSBase` and implements the core methods:

* **`run_mcts`** : Recursively performs the MCTS search using DNG-based uncertainty estimates.
* **`expansion`** : Expands the current node by adding child nodes based on possible actions.
* **`selection`** : Selects the best child node using exploration constants and sampling.
* **`test_episode`** : Evaluates the policy by running a test episode in the environment.

### `RaceTrack` Environment

Currently, The agent learns in the `RaceTrack` environment which simulates a racing scenario where the agent must navigate a track using the learned policy. The environment dynamics can be stochastic, making it a challenging test case for MCTS algorithms. The implementation for the `RaceTrack` environment is from 



## 💾 Checkpointing

During training, checkpoints are saved at regular intervals in the `checkpoints/` directory. To load a checkpoint for evaluation, ensure the corresponding `.pkl` file is present in the directory. The evaluation script will load the checkpoint and replay the stored policy.



## 📈 Results

The DNG-MCTS algorithm shows improved performance in environments with stochastic dynamics due to its uncertainty modeling using the Dirichlet Normal Gamma distribution. It effectively balances exploration and exploitation, reducing the need for extensive parameter tuning.


## 📝 References

* [DNG-MCTS Paper: NeurIPS 2013](https://proceedings.neurips.cc/paper/2013/hash/846c260d715e5b854ffad5f70a516c88-Abstract.html)
* Sutton, R. S., & Barto, A. G. (2018).  *Reinforcement Learning: An Introduction* .
