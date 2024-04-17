# Reinforcement Learning for Pong

Implementations of reinforcement learning models to play Pong in Python from scratch. The game is built in Rust with a PyO3 interface exposed to Python to export the game state and frame data. All models are implemented using NumPy and saved as JSON files.

With a relatively simple neural network architecture, the agent learns to play the game and defeats the hard-coded computer opponent about 93% of the time for the Deep Q-Learning model and about 86% of the time for the vanilla REINFORCE policy gradient model. More detailed measurements and results can be found in `agent/results.md`.

Implementation of Pong environment: `src` folder
Implementation of RL agents: `agent`

https://github.com/MrEconomical/pong-rl/assets/47700125/6f099ef0-3ab8-40d9-a3ed-7a753feaefc4