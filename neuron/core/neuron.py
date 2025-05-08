import numpy as np
import json


class Neuron:
    def __init__(self, input_size, activation=None):
        if activation is None:
            self.activation = lambda x: np.where(x >= 0, 1, 0)
        else:
            self.activation = activation

        self.weights = np.zeros(input_size, dtype=np.float16)
        self.bias = 0.0

    def forward(self, inputs):
        inputs = np.asarray(inputs)
        z = np.dot(self.weights, inputs) + self.bias
        return self.activation(z)

    def update_weights(self, inputs, error, learning_rate):
        inputs = np.asarray(inputs)
        delta = learning_rate * error * inputs
        self.weights += delta
        self.bias += learning_rate * error

    def save_weights(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({
                'weights': self.weights.tolist(),
                'bias': float(self.bias)
            }, f)

    def load_weights(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.weights = np.array(data['weights'], dtype=np.float64)
            self.bias = data['bias']
