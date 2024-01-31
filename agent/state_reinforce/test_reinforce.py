'''
test REINFORCE policy gradient model
'''

from pathlib import Path
import sys
sys.path.insert(0, str(Path(Path(__file__).parent.absolute()).parent.absolute()))

from models.policy_model import Model
import numpy as np
import pong_rl

checkpoint = 14
save_folder = "reinforce_models_1"
model = Model.from_save("agent/state_reinforce/" + save_folder + "/" + str(checkpoint) + ".json")
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
        game_state = pong.get_normalized_state()
        h, action_probs = model.forward(game_state)
        action = np.random.choice(action_probs.size, p=action_probs)
        print(action_probs, action)

        reward = pong.tick(action)
        if reward == 0:
            reward = pong.tick(action)
    
    pong.reset()