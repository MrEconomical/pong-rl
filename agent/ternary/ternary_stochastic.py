from pathlib import Path
import sys
sys.path.insert(0, str(Path(Path(__file__).parent.absolute()).parent.absolute()))

from models.stochastic_model import Model
import random

# ternary inputs and outputs

ternary_cases = [
    ([0, 0, 0], [0, 1]),
    ([0, 0, 1], [1, 0]),
    ([0, 1, 0], [0, 1]),
    ([0, 1, 1], [1, 0]),
    ([1, 0, 0], [0, 1]),
    ([1, 0, 1], [0, 1]),
    ([1, 1, 0], [1, 0]),
    ([1, 1, 1], [1, 0]),
]

# train model over epochs

input_size = 3
hidden_size = 6
output_size = 2
learning_rate = 0.02
epochs = 20000
model = Model.with_random_weights(input_size, hidden_size, output_size, learning_rate)

log_interval = epochs // 10
for e in range(epochs):
    total_error = 0
    train_cases = ternary_cases.copy()
    random.shuffle(train_cases)

    for case in train_cases:
        hidden_output, output = model.forward(case[0])
        output_error = model.back_prop(case[0], hidden_output, output, case[1])
        total_error += output_error
    
    if (e + 1) % log_interval == 0:
        print("epoch", e + 1, "mean error", total_error / len(train_cases))

print()
print("evaluating model:")
for case in ternary_cases:
    h, output = model.forward(case[0])
    print("output", output, "expected", case[1])