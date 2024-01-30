'''
REINFORCE policy gradient with full state of Pong environment
'''

from pathlib import Path
import sys
sys.path.insert(0, str(Path(Path(__file__).parent.absolute()).parent.absolute()))

from models.policy_model import Model
import numpy as np
import pong_rl

# create or load model

save_folder = "reinforce_models"
load_model = False
checkpoint = 0
log_interval = 1000
save_interval = 1000
print("save folder: " + save_folder)

model = None
if load_model:
    model = Model.from_save("agent/state_reinforce/" + save_folder + "/" + str(checkpoint) + ".json")
    print("loaded model with parameters ({}, {}, {}, {}) from checkpoint {}".format(
        model.input_size,
        model.hidden_size,
        model.learning_rate,
        model.discount_rate,
        checkpoint,
    ))
else:
    model = Model.with_random_weights(
        6, # input size
        300, # hidden size
        2, # output size
        0.001, # learning rate
        0.99, # discount rate
    )
    print("created new model with parameters ({}, {}, {}, {})".format(
        model.input_size,
        model.hidden_size,
        model.learning_rate,
        model.discount_rate,
    ))

# create Pong environment

pong = pong_rl.PongEnv.without_render()
episode_num = 0
wins = 0
losses = 0