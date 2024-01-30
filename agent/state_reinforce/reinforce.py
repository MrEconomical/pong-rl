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

save_folder = "reinforce_models_1"
load_model = False
checkpoint = 9
log_interval = 2000
save_interval = 2000
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

# initialize training data

hidden_batch = np.zeros((model.hidden_size, model.input_size + 1))
output_batch = np.zeros((model.output_size, model.hidden_size + 1))
batch_size = 32

while True:
    # initialize episode data

    pong.start()
    episode_num += 1
    episode_states = []
    episode_hidden_outputs = []
    episode_outputs = []
    episode_actions = []
    final_reward = 0

    while final_reward == 0:
        # predict action

        game_state = pong.get_normalized_state()
        hidden_output, action_probs = model.forward(game_state)
        action = np.random.choice(action_probs.size, p=action_probs)

        # store state and action data

        episode_states.append(game_state)
        episode_hidden_outputs.append(hidden_output)
        episode_outputs.append(action_probs)

        action_vector = np.zeros(action_probs.size)
        action_vector[action] = 1
        episode_actions.append(action_vector)

        # advance game state

        final_reward = pong.tick(action)
        if final_reward == 0:
            final_reward = pong.tick(action)

    # calculate discounted rewards
            
    episode_rewards = np.empty(len(episode_states))
    episode_rewards[len(episode_states) - 1] = final_reward
    for s in range(len(episode_states) - 2, -1, -1):
        episode_rewards[s] = episode_rewards[s + 1] * model.discount_rate

    episode_rewards -= np.mean(episode_rewards)
    episode_rewards /= np.std(episode_rewards)
    
    # calculate policy gradients

    for s in range(len(episode_states)):
        hidden_grad, output_grad = model.back_prop(
            episode_states[s],
            episode_hidden_outputs[s],
            episode_outputs[s],
            episode_actions[s],
            episode_rewards[s],
        )

        hidden_batch += hidden_grad
        output_batch += output_grad
    
    # apply gradient batch

    if episode_num % batch_size == 0:
        model.apply_gradients(hidden_batch, output_batch)
        hidden_batch.fill(0)
        output_batch.fill(0)

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