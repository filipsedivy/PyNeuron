from neuron.layers.perceptron import Perceptron
from neuron.utils.callbacks import EarlyStopping

import numpy as np

# logical AND function
# inputs: [x1, x2]
# outputs: x1 AND x2

training_inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)

labels = np.array([0, 0, 0, 1], dtype=np.float32)  # Target values (AND)

input_dimension = training_inputs.shape[1]
learning_rate = 0.1
epochs = 50

# Initialize Perceptron
perceptron = Perceptron(input_dimension, learning_rate)

print("Initial weights:", perceptron.neuron.weights, "Bias:", perceptron.neuron.bias)

# Training
early_stopping = EarlyStopping(patience=20)
perceptron.train(training_inputs, labels, epochs=epochs, callbacks=[early_stopping])

print("\nTraining finished.")
print("Final weights:", perceptron.neuron.weights, "Bias:", perceptron.neuron.bias)

# Testing on training data (batch prediction)
print("\nTesting on training data:")
predictions = perceptron.predict(training_inputs)
print("Inputs:\n", training_inputs)
print("Predictions (batch):", predictions)
print("Expected outputs:", labels)

# Testing on single data points (predict handles single sample now)
print("\nTesting on single data points:")
print(f"Input [0, 0]: {perceptron.predict(np.array([0, 0], dtype=np.float32))}")
print(f"Input [0, 1]: {perceptron.predict(np.array([0, 1], dtype=np.float32))}")
print(f"Input [1, 0]: {perceptron.predict(np.array([1, 0], dtype=np.float32))}")
print(f"Input [1, 1]: {perceptron.predict(np.array([1, 1], dtype=np.float32))}")

# Example usage of save/load
file_path = "perceptron_weights.json"
print(f"\nSaving weights to {file_path}")
perceptron.save(file_path)

# Create a new instance and load weights
new_perceptron = Perceptron(input_dimension)  # Weights will be random upon initialization
print("New instance, initial weights:", new_perceptron.neuron.weights)

print(f"Loading weights from {file_path}")
new_perceptron.load(file_path, input_dimension)  # Loads weights from file
print("New instance, loaded weights:", new_perceptron.neuron.weights)

# Test the loaded model
print("\nTesting the loaded model:")
predictions_loaded = new_perceptron.predict(training_inputs)
print("Predictions (loaded model):", predictions_loaded)