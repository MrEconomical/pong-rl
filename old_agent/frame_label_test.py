from models.mse_model import Model
import numpy as np
import pong_rl

checkpoint = 7
model = Model.from_save("agent/saved_models/frame_label/" + str(checkpoint) + ".json")
print("loaded model with parameters ({}, {}, {}) from checkpoint {}".format(
    model.input_size,
    model.hidden_size,
    model.learning_rate,
    checkpoint,
))

pong = pong_rl.PongEnv.with_render()
pong.start()

while True:
    reward = 0
    while reward == 0:
        frame_state = pong.get_normalized_frame()
        h, action_prob = model.forward(frame_state)
        action = 1 if np.random.uniform() < action_prob else 0
        reward = pong.tick(action)
    pong.reset()
