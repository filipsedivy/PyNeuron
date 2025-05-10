import numpy as np
import json

from typing import Callable, Optional


class Neuron:
    def __init__(self,
                 input_size: int,
                 activation: Optional[Callable] = None,
                 weights: Optional[np.ndarray] = None,
                 bias: float = 0.0):
        if activation is None:
            self.activation = lambda x: np.where(x >= 0, 1, 0)
        else:
            self.activation = activation

        if weights is None:
            limit = np.sqrt(6 / (input_size + 1))
            self.weights = np.random.uniform(-limit, limit, input_size).astype(np.float32)
        elif isinstance(weights, np.ndarray):
            if weights.shape != (input_size,):
                raise ValueError(f"Invalid weights shape: {weights.shape}, expected {(input_size,)}")
            self.weights = weights.astype(np.float32)
        else:
            raise TypeError(f"Invalid weights type: {type(weights)}")

        self.bias = np.float32(bias)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.dtype != np.float32:
            inputs = inputs.astype(np.float32)

        z = np.dot(self.weights, inputs) + self.bias
        return self.activation(z)

    def update_weights(self, inputs: np.ndarray, error: float, learning_rate: float) -> None:
        if inputs.dtype != np.float32:
            inputs = inputs.astype(np.float32)

        delta = learning_rate * error * inputs
        self.weights += delta
        self.bias += np.float32(learning_rate * error)

    def save_weights(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({
                'weights': self.weights.tolist(),
                'bias': float(self.bias)
            }, f)

    def load_weights(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.weights = np.array(data['weights'], dtype=np.float32)
            self.bias = np.float32(data['bias'])

    def batch_forward(self, batch_inputs: np.ndarray) -> np.ndarray:
        if batch_inputs.dtype != np.float32:
            batch_inputs = batch_inputs.astype(np.float32)

        if len(batch_inputs.shape) == 1:
            batch_inputs = batch_inputs.reshape(1, -1)

        if batch_inputs.shape[1] != len(self.weights):
            raise ValueError(f"Input dimensions don't match: expected {len(self.weights)}, got {batch_inputs.shape[1]}")

        z = np.dot(batch_inputs, self.weights) + self.bias
        return self.activation(z)

    def batch_update_weights(self, batch_inputs: np.ndarray, batch_errors: np.ndarray, learning_rate: float) -> None:
        if batch_inputs.dtype != np.float32:
            batch_inputs = batch_inputs.astype(np.float32)

        if batch_errors.dtype != np.float32:
            batch_errors = batch_errors.astype(np.float32)

        if len(batch_inputs.shape) == 1:
            batch_inputs = batch_inputs.reshape(1, -1)

        if len(batch_errors.shape) == 0:
            batch_errors = np.array([batch_errors])

        mean_delta = np.mean(batch_errors[:, np.newaxis] * batch_inputs, axis=0) * learning_rate
        self.weights += mean_delta
        self.bias += np.float32(np.mean(batch_errors) * learning_rate)
