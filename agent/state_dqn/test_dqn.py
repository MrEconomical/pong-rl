'''
test state Deep Q-Network model
'''

from pathlib import Path
import sys
sys.path.insert(0, str(Path(Path(__file__).parent.absolute()).parent.absolute()))

from models.dqn_model import Model
import pong_rl

checkpoint = 24
save_folder = "dqn_models"
model = Model.from_save("agent/state_dqn/" + save_folder + "/" + str(checkpoint) + ".json")
print("loaded model with parameters ({}, {}, {}, {}, {}, {}) from checkpoint {}".format(
    model.input_size,
    model.hidden_size,
    model.output_size,
    model.learning_rate,
    model.discount_rate,
    model.explore_factor,
    checkpoint,
))

pong = pong_rl.PongEnv.with_render()
pong.start()

while True:
    reward = 0
    
    while reward == 0:
        game_state = pong.get_normalized_state()
        h, action_values = model.forward(game_state)
        action = 0 if action_values[0] >= action_values[1] else 1
        print("action values:", action_values)

        reward = pong.tick(action)
        if reward == 0:
            reward = pong.tick(action)
    
    pong.reset()