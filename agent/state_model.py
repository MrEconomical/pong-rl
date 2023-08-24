# reinforcement learning agent with full state of Pong environment

import numpy as np
import pong_rl
import random

PongEnv = pong_rl.PongEnv

pong = PongEnv.with_render()
pong.start()
while True:
    reward = pong.tick(0 if random.random() < 0.5 else 1)
    game_state = pong.get_game_state()
    print("game state:", game_state)
    if reward != 0:
        print("final reward:", reward)
        break