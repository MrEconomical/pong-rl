'''
test frame label batch model
'''

from pathlib import Path
import sys
sys.path.insert(0, str(Path(Path(__file__).parent.absolute()).parent.absolute()))

from models.batch_model import Model
import numpy as np
import pong_rl

checkpoint = 8
save_folder = "batch_models"
model = Model.from_save("agent/frame_label/" + save_folder + "/" + str(checkpoint) + ".json")
print("loaded model with parameters ({}, {}, {}, {}) from checkpoint {}".format(
    model.input_size,
    model.hidden_size,
    model.output_size,
    model.learning_rate,
    checkpoint,
))

pong = pong_rl.PongEnv.with_render()
pong.start()

while True:
    prev_frame = pong.get_normalized_frame()
    reward = 0

    while reward == 0:
        current_frame = pong.get_normalized_frame()
        stacked_frame = np.concatenate((prev_frame, current_frame))
        prev_frame = current_frame

        h, action_prob = model.forward(stacked_frame)
        action = 1 if np.random.uniform() < action_prob[0] else 0

        reward = pong.tick(action)
        if reward == 0:
            reward = pong.tick(action)
    
    pong.reset()