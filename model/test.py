import pong_rl

PongEnv = pong_rl.PongEnv

pong = PongEnv.with_render()
pong.start()
while True:
    pass