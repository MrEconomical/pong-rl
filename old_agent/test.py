# labeled learning with full state of Pong environment

from models.test_model import Model
import numpy as np
import pong_rl

# create or load model

load_model = False
checkpoint = 0

model = None
if load_model:
    model = Model.from_save("agent/saved_models/state_label/" + str(checkpoint) + ".json")
    print("loaded model with parameters ({}, {}, {}) from checkpoint {}".format(
        model.input_size,
        model.hidden_size,
        model.learning_rate,
        checkpoint,
    ))
else:
    model = Model.with_random_weights(
        6, # input size
        20, # hidden size
        0.001, # learning rate
    )
    print("created new model with parameters ({}, {}, {})".format(
        model.input_size,
        model.hidden_size,
        model.learning_rate,
    ))

# create Pong environment

pong = pong_rl.PongEnv.without_render()
episode_num = 0
wins = 0
losses = 0

while True:
    hidden_buffer = np.zeros((model.hidden_size, model.input_size + 1))
    output_buffer = np.zeros(model.hidden_size + 1)

    for i in range(100):
        # initialize episode data

        pong.start()
        episode_num += 1
        episode_states = []
        episode_hidden_outputs = []
        episode_probs = []
        episode_labels = []
        final_reward = 0

        while final_reward == 0:
            # predict action

            game_state = pong.get_normalized_state()
            hidden_output, action_prob = model.forward(game_state)
            action = 1 if np.random.uniform() < action_prob else 0

            # calculate correct action

            #correct_action = None
            #if game_state[1] < game_state[4]:
            #    correct_action = 1
            #else:
            #    correct_action = 0
            correct_action = action

            # store action data and get reward

            episode_states.append(game_state)
            episode_hidden_outputs.append(hidden_output)
            episode_probs.append(action_prob)
            episode_labels.append(correct_action)

            final_reward = pong.tick(action)
            if final_reward == 0:
                final_reward = pong.tick(action)
        
        if final_reward == -1:
            losses += 1
        else:
            wins += 1
        
        # back propagate labels through model
        
        for s in range(len(episode_states)):
            actual_label = None
            if final_reward == -1:
                if episode_labels[s] == 1:
                    actual_label = 0
                else:
                    actual_label = 1
            else:
                actual_label = episode_labels[s]

            hidden_gradients, output_gradient = model.back_prop(
                episode_states[s],
                episode_hidden_outputs[s],
                episode_probs[s],
                actual_label,
            )
            hidden_buffer += hidden_gradients
            output_buffer += output_gradient

        # reset game environment

        pong.reset()
    
    model.update_gradients(hidden_buffer, output_buffer)
    
    if episode_num % 10000 == 0:
        print("FINISHED EPISODE:", episode_num)
        print("wins and losses:", wins, losses)
        checkpoint += 1
        model.save("agent/saved_models/test/" + str(checkpoint) + ".json")