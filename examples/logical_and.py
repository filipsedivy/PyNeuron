from neuron.layers.perceptron import Perceptron
from neuron.utils.callbacks import EarlyStopping

# logical AND function
# inputs: [x1, x2]
# outputs: x1 AND x2

training_inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

outputs = [0, 0, 0, 1]

print("Train perceptron...")

early_stopping = EarlyStopping(patience=5)

perceptron = Perceptron(input_size=2, learning_rate=0.1)
perceptron.train(training_inputs, outputs, epochs=10, callbacks=[early_stopping])
perceptron.neuron.save_weights("weight.json")

print("Predictions after training:")
prediction = Perceptron(input_size=2, learning_rate=0.1)
prediction.neuron.load_weights("weight.json")

print(f"Neural network has {prediction.count_parameters()} parameters.")

for inputs in training_inputs:
    output = prediction.predict(inputs)
    print(f"Input: {inputs} => Output: {output}")
