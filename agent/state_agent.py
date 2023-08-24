# reinforcement learning agent with full state of Pong environment

from stochastic_model import Model
import numpy as np
import pong_rl
import random

# create model

model = Model(
    3, # input size
    5, # hidden size
    0.05 # learning rate
)

# create Pong environment

pong = pong_rl.PongEnv.with_render()
pong.start()

while True:
    reward = pong.tick(0 if random.random() < 0.5 else 1)
    game_state = pong.get_game_state()
    print("game state:", game_state)
    if reward != 0:
        print("final reward:", reward)
        break