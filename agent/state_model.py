# reinforcement learning agent with full state of Pong environment

from model import Model
import numpy as np
import pong_rl
import random

PongEnv = pong_rl.PongEnv
model = Model(
    3, # input size
    5, # hidden size
    0.05 # learning rate
)

ternary_cases = [
    ([0, 0, 0], 0),
    ([0, 1, 0], 0),
    ([0, 0, 1], 1),
    ([0, 1, 1], 1),
    ([1, 0, 0], 0),
    ([1, 1, 0], 1),
    ([1, 0, 1], 0),
    ([1, 1, 1], 1),
]

for x in range(50000):
    average_loss = 0
    for case in ternary_cases:
        hidden_output, output = model.forward(case[0])
        loss = model.back_prop(case[0], hidden_output, output, case[1])
        average_loss += loss
    average_loss /= len(ternary_cases)
    if x % 10000 == 0:
        print("mean loss:", average_loss)

print(model.weights)
print("testing network:")
for case in ternary_cases:
    h, output = model.forward(case[0])
    print("expected:", case[1], "predicted:", output)

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