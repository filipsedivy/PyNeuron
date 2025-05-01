from ..core.neuron import Neuron


class Perceptron:
    def __init__(self, input_size, learning_rate=1.0):
        self.neuron = Neuron(input_size)
        self.learning_rate = learning_rate

    def predict(self, inputs):
        return self.neuron.forward(inputs)

    def train(self, training_data, labels, epochs=10, callbacks=None):
        if callbacks is None:
            callbacks = []

        for epoch in range(epochs):
            total_error = 0
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                total_error += abs(error)
                self.neuron.update_weights(inputs, error, self.learning_rate)

            print(f"Epoch {epoch + 1}/{epochs} - Total Error: {total_error}")

            for callback in callbacks:
                callback(epoch, total_error)
                if hasattr(callback, "stopped") and callback.stopped:
                    return
