# JaxLife - An Open-Ended Artificial Life Simulator in Jax

## 🧬 What is JaxLife
JaxLife is an artificial life simulator that focuses on evolving complex and high-level behaviours. 
This is done by having agents interact with robot systems that can perform Turing-complete computation.


### 💻 Simulation
Our simulation consists of *agents*, *robots* and *terrain*. Agents evolve via natural selection, and are parametrised by a recurrent neural network. Robots can be programmed by agents, and are able to perform Turing-complete computation.
Finally, the terrain controls many aspects of the simulation, such as how easy it is to move, how much energy there is available, etc. Terrain also changes slowly due to a weather and climate-like system.
<p align="center">
    <img src="pics/fig1.png">
</p>

### 👾 Agents
The agent's architecture uses separate encoders for all nearby entities. These are processes using self attention, and then cross attention with the agent's own embedding. The terrain is processed by a different encoder, and concatenated with the final entity representation. All of this is passed through an LSTM.

Agents evolve via natural selection, and pass on their weights to their offspring, with small random perturbations acting as mutations.
Agents can perform several actions, such as moving, eating, terraforming, attacking, and programming robots.
<p align="center">
    <img src="pics/agents.png">
</p>

### 🤖 Robots
Robots have the same action space as agents and can be programmed by sending a message to them. 
These robots can theoretically perform Turing-complete computation.
<p align="center">
    <img src="pics/rule110.gif">
</p>

But they are also able to perform useful tasks, such as transportation, farming and communication.

<p align="center">
    <img width="30%" src="pics/train.gif">
    <img width="30%" src="pics/oscillate.gif">
    <img width="30%" src="pics/terraform.gif">
</p>

## 📝 Results
When we run the simulation, we can see some interesting emergent properties, such as the agents and robots performing mass terraforming and coordinating their actions.
<p align="center">
    <img src="pics/bridge.gif">
</p>

## ✍️ Usage

### 🛠️ Installation

```bash
git clone https://github.com/luchris429/JaxLife.git
cd JaxLife
pip install -r requirements.txt
pre-commit install
```

### 🏃 Running
You can run the following to start a simulation. There are many different configuration options, see the `src/main.py` for more details. 
A few important parameters are:
- `--gui`: Run interactively using Pygame
- `--wandb`: Whether or not to log to Weights and Biases.
- `--num-agents`: How many agents are in the simulation.
```bash
python src/main.py
```


## 🔍 See Also

- [Alien](https://github.com/chrxh/alien): A CUDA-powered Alife simulation.
- [Leniax](https://github.com/morgangiraud/leniax): [Lenia](https://chakazul.github.io/lenia.html#Code) in Jax