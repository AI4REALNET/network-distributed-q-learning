# Network-distributed Q-learning

#### Short description of the algorithm
The Network-distributed Q-learning algorithm is a distributed version of the popular [Q-Learning algorithm](https://link.springer.com/article/10.1007/BF00992698).

The original update of state-value function of an agent according to the Q-learning algorithm is:

$$
Q(s,a) = (1-\alpha) Q(s,a) + \alpha(r + \gamma\max_{a'} Q(s,a'))
$$

The distributed version instead has the following update for each agent:

$$
Q(s,a) = (1-\alpha) Q(s,a) + \alpha(r + \gamma\max_{a'} Q_{\text{next}}(s,a'))
$$

where $Q_{\text{next}}$ is the state-value function of the successor agent, i.e., the successor node in the graph.

The algorithm is tested on SwitchFL, in which each junction cell is modeled as a node in the graph. Each node is an independent RL agent that exchanges information with the successor node, i.e., where the train is sent starting from the current node.



#### SwitchFL
[**SwitchFL**](https://github.com/gvlos/SwitchFL) is a custom **multi-agent reinforcement learning (MARL)** environment based on [Flatland](https://flatland.aicrowd.com/getting-started/env.html), designed with a novel **switch-centered** perspective. This environment shifts the focus of control from individual trains to **railway switches**, introducing unique coordination and planning challenges in a rail network.


## Motivation

Traditional train-routing environments like Flatland focus on agent-centered control. **SwitchFL** introduces a novel abstraction by modeling **switches** as decision points. This perspective is better suited for **decentralized control**, **real-world railway signal systems**, and **asynchronous** agent interactions. 

It’s particularly useful for:
- Studying coordination across control points in transportation systems.
- Training RL agents in asynchronous, partially observable environments.
- Benchmarking switch-based vs. agent-based control.


#### Overview of code structure
:open_file_folder: **network-distributed q-learning**

├── :open_file_folder: flatland patch

│   └── ...

├── :open_file_folder: switchfl

│   └── ...

├── eval.py

├── hyperparam_tuning.py

├── main.py

├── plot.ipynb

├── test_model.py


The folder *flatland_patch* contains the patch to the Flatland environment, the folder *switchfl* contains the implementation of the SwitchFL environment and the network-distributed Q-learning algorithm. The python scripts *main.py*, *eval.py*, *hyperaparam_tuning.py*, *test_model.py*, *plot.py* can be used to train and test the algorithm, plot the figures of the experiments.


## Installation

Create a virtual environment and install all the dependencies

```bash
python -m venv <my_venv>
source <my_venv>/bin/activate
pip install -r requirements.txt
```

Apply the patch to Flatland. In the folder `<my_venv>/lib/python3.x/site-packages/flatland/envs` replace the files contained in the folder *flatland_patch*.

## Usage Example (Input/Output)

You can test the library by runnning the file *test_model.py*. Edit the environment configuration variables and run the script. The result will be a folder with the following content

- *arrived_trains.npz* contains the number of the trains that have reached the target station (for each training episode)
- *arrived_trains_exploit.npz* contains the number of trains that have reached the target station with the greedy policy (for each exploit checkpoint)
- *cum_reward.npz* contains the cumulative rewards over the training episodes
- *cum_reward_exploit.npz* contains the cumulative rewards over the training episodes by using the greedy policy
- *delays.npz* contains the final delays for each train
- *distr_q_model.pkl* contains the Q-tables of the agents
- *num_malfunctions.npz* contains the number of malfunctions happened over the training episodes
- *trains_at_dest.npz* contains the ID of the trains that have reached the target station


You can run a single or multiple experiment with the file *hyperparam_tuning.py*. Experiments will be run in parallel.
A folder will be created with one subfolder for each parameter configuration. Inside each subfolder there will be other subfolders corresponding to the different runs of the experiments (different seeds). The content of each such folder is analogous to the content explained before.


## Reproducing experiments

In order to reproduce the experiment presented in the slides you can run the file *hyperparam_tuning* as is and use the *plot.ipynb* file to plot the same curves. For the evaluation plot you need first to run *eval.py* by specifying the set of experiment directories to evaluate and the name of the RL model. It will create a subfolder *eval_x* with three files that can be used for plotting (*cum_rewards.npy*, *delays.npz*, *trains_at_dest.npz*)