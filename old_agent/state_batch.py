# batch gradient descent with full state of Pong environment

from models.batch_model import Model
import numpy as np
import pong_rl

# create or load model

load_model = True
checkpoint = 11
epoch_length = 100

model = None
if load_model:
    model = Model.from_save("agent/saved_models/state_batch/" + str(checkpoint) + ".json")
    print("loaded model with parameters ({}, {}, {}) from checkpoint {}".format(
        model.input_size,
        model.hidden_size,
        model.learning_rate,
        checkpoint,
    ))
else:
    model = Model.with_random_weights(
        6, # input size
        100, # hidden size
        0.99, # reward discount rate
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
    # collect and train agent over epoch

    hidden_gradients = np.zeros((model.hidden_size, model.input_size + 1))
    output_gradient = np.zeros(model.hidden_size + 1)

    for e in range(epoch_length):
        # initialize epoch data

        pong.start()
        episode_num += 1
        episode_states = []
        episode_hidden_outputs = []
        episode_probs = []
        final_reward = 0
        
        while final_reward == 0:
            # predict action

            game_state = pong.get_normalized_state()
            hidden_output, action_prob = model.forward(game_state)
            action = 1 if np.random.uniform() < action_prob else 0

            # store action data and get reward

            episode_states.append(game_state)
            episode_hidden_outputs.append(hidden_output)
            episode_probs.append(action - action_prob)
            final_reward = pong.tick(action)
        
        if final_reward == -1:
            losses += 1
        else:
            wins += 1
        
        # back propagate discounted rewards through model and add gradients
        
        discounted_rewards = model.discount_rewards(final_reward, len(episode_states))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        hidden_update, output_update = model.back_prop(
            np.array(episode_states),
            np.array(episode_hidden_outputs),
            np.array(episode_probs),
            discounted_rewards,
        )
        hidden_gradients += hidden_update
        output_gradient += output_update
        
        # reset game environment

        pong.reset()
    
    # apply total gradients to update weights

    model.apply_gradients(hidden_gradients, output_gradient)

    if episode_num % 5000 == 0:
        print("FINISHED EPISODE:", episode_num)
        print("wins and losses:", wins, losses)
        print(model.weights[1][0:5])
    if episode_num % 50000 == 0:
        checkpoint += 1
        model.save("agent/saved_models/state_batch/" + str(checkpoint) + ".json")