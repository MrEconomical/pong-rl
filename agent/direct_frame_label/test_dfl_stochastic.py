'''
test direct frame label stochastic model
'''

from pathlib import Path
import sys
sys.path.insert(0, str(Path(Path(__file__).parent.absolute()).parent.absolute()))

from models.stochastic_model import Model
import numpy as np
import pong_rl

checkpoint = 8
save_folder = "stochastic_models"
model = Model.from_save("agent/direct_frame_label/" + save_folder + "/" + str(checkpoint) + ".json")
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
    reward = 0

    while reward == 0:
        current_frame = pong.get_normalized_frame()
        h, action_prob = model.forward(current_frame)
        action = 1 if np.random.uniform() < action_prob[0] else 0

        reward = pong.tick(action)
        if reward == 0:
            reward = pong.tick(action)
    
    pong.reset()
