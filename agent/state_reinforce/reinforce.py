'''
REINFORCE policy gradient with full state of Pong environment
'''

from pathlib import Path
import sys
sys.path.insert(0, str(Path(Path(__file__).parent.absolute()).parent.absolute()))

from models.array_vec import ArrayVec
from models.policy_model import Model
import numpy as np
import pong_rl

# create or load model

save_folder = "reinforce_models"
load_model = False
checkpoint = 0
log_interval = 8000
save_interval = 8000
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
        600, # hidden size
        2, # output size
        0.0002, # learning rate
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

# initialize training data

batch_size = 400
initial_len = 80000
extend_len = 20000

batch_states = ArrayVec((model.input_size,), initial_len, extend_len)
batch_hidden_outputs = ArrayVec((model.hidden_size,), initial_len, extend_len)
batch_outputs = ArrayVec((model.output_size,), initial_len, extend_len)
batch_actions = ArrayVec((model.output_size,), initial_len, extend_len)
batch_rewards = []

while True:
    # initialize episode data

    pong.start()
    episode_num += 1
    num_states = 0
    final_reward = 0

    while final_reward == 0:
        # predict action

        game_state = pong.get_normalized_state()
        hidden_output, action_probs = model.forward(game_state)
        action = np.random.choice(action_probs.size, p=action_probs)

        # store state and action data

        num_states += 1
        batch_states.push(game_state)
        batch_hidden_outputs.push(hidden_output)
        batch_outputs.push(action_probs)

        action_vector = np.zeros(action_probs.size)
        action_vector[action] = 1
        batch_actions.push(action_vector)

        # advance game state

        final_reward = pong.tick(action)
        if final_reward == 0:
            final_reward = pong.tick(action)

    # calculate discounted rewards

    for s in range(num_states):
        reward = final_reward * (model.discount_rate ** (num_states - s - 1))
        batch_rewards.append(reward)

    if episode_num % batch_size == 0:
        # normalize batch rewards

        batch_rewards = np.array(batch_rewards)
        batch_rewards -= np.mean(batch_rewards)
        batch_rewards /= np.std(batch_rewards)

        # calculate and apply policy gradients

        hidden_grads, output_grads = model.batch_back_prop(
            batch_states.get_ref(),
            batch_hidden_outputs.get_ref(),
            batch_outputs.get_ref(),
            batch_actions.get_ref(),
            batch_rewards,
        )
        model.apply_gradients(
            np.sum(hidden_grads, axis=0),
            np.sum(output_grads, axis=0),
        )

        # reset batch data

        batch_states.clear()
        batch_hidden_outputs.clear()
        batch_outputs.clear()
        batch_actions.clear()
        batch_rewards = []

    # update stats counter
    
    if final_reward == -1:
        losses += 1
    else:
        wins += 1

    # reset game environment
        
    pong.reset()
    
    if episode_num % log_interval == 0:
        print("FINISHED EPISODE:", episode_num)
        print("wins and losses:", wins, losses)
        wins = 0
        losses = 0
    
    if episode_num % save_interval == 0:
        checkpoint += 1
        model.save("agent/state_reinforce/" + save_folder + "/" + str(checkpoint) + ".json")