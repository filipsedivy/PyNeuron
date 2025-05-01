from ..utils.linalg import dot_product, scalar_multiply, vector_add
from ..functions.activations import step_function

import json


class Neuron:
    def __init__(self, input_size, activation=step_function):
        self.weights = [0.0] * input_size
        self.bias = 0.0
        self.activation = activation

    def forward(self, inputs):
        z = dot_product(self.weights, inputs) + self.bias
        return self.activation(z)

    def update_weights(self, inputs, error, learning_rate):
        delta = scalar_multiply(learning_rate * error, inputs)
        self.weights = vector_add(self.weights, delta)
        self.bias += learning_rate * error

    def save_weights(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({
                'weights': self.weights,
                'bias': self.bias
            }, f)

    def load_weights(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.weights = data['weights']
            self.bias = data['bias']
