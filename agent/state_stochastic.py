# stochastic gradient descent with full state of Pong environment

from stochastic_model import Model
import numpy as np
import pong_rl
import random

# create model

model = Model.with_random_weights(
    6, # input size
    10, # hidden size
    0.01 # learning rate
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