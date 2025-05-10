from ..core.neuron import Neuron
from typing import Optional, Callable, List
import numpy as np


class Perceptron:
    def __init__(self, input_size: int, learning_rate: float = 1.0):
        self.neuron = Neuron(input_size)
        self.learning_rate = learning_rate

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.neuron.batch_forward(inputs)

    def train(self,
              training_data: np.ndarray,
              labels: np.ndarray,
              epochs: int = 10,
              callbacks: Optional[List[Callable]] = None) -> None:
        if callbacks is None:
            callbacks = []

        if not isinstance(training_data, np.ndarray) or training_data.dtype != np.float32:
            training_data = np.array(training_data, dtype=np.float32)

        if not isinstance(labels, np.ndarray) or labels.dtype != np.float32:
            labels = np.array(labels, dtype=np.float32)

        num_samples = training_data.shape[0]

        print(f"Starting training for {epochs} epochs with {num_samples} samples...")

        for epoch in range(epochs):
            predictions = self.neuron.batch_forward(training_data)
            batch_errors = labels - predictions
            total_epoch_error = np.sum(np.abs(batch_errors))
            self.neuron.batch_update_weights(training_data, batch_errors, self.learning_rate)

            print(f"Epoch {epoch + 1}/{epochs} - Total Error: {total_epoch_error:.4f}")

            for callback in callbacks:
                callback(epoch, total_epoch_error)
                if hasattr(callback, "stopped") and callback.stopped:
                    print(f"Training stopped early by callback after epoch {epoch + 1}")
                    return

    def count_parameters(self) -> int:
        weights_count = len(self.neuron.weights)
        total_params = weights_count + 1  # one for bias
        return total_params

    def save(self, filepath: str):
        self.neuron.save_weights(filepath)

    def load(self, filepath: str, input_size: int):
        if hasattr(self, "neuron"):
            if len(self.neuron.weights) != input_size:
                raise ValueError(
                    f"Model mismatch: Loaded weights are for input size {input_size}, but existing Neuron expects {len(self.neuron.weights)}")
        else:
            self.neuron = Neuron(input_size)

        self.neuron.load_weights(filepath)
