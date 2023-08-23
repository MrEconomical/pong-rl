import pong_rl
import random

PongEnv = pong_rl.PongEnv

pong = PongEnv.with_render()
pong.start()
while True:
    reward = pong.tick(0 if random.random() < 0.5 else 1)
    if reward != 0:
        print("final reward:", reward)
        break