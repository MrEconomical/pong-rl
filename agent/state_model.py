# reinforcement learning agent with full state of Pong environment

from model import Model
import numpy as np
import pong_rl
import random

PongEnv = pong_rl.PongEnv
model = Model(
    6, # input size
    20, # hidden size
    0.01 # learning rate
)
print(model.weights)

'''
pong = PongEnv.with_render()
pong.start()
while True:
    reward = pong.tick(0 if random.random() < 0.5 else 1)
    game_state = pong.get_game_state()
    print("game state:", game_state)
    if reward != 0:
        print("final reward:", reward)
        break
'''