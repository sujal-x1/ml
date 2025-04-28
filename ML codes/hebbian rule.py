import numpy as np

class HebbianNetwork:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        self.weights = np.random.rand(input_size, output_size)
        self.learning_rate = learning_rate

    def train(self, inputs, outputs):
        for i in range(len(inputs)):
            input_vector = inputs[i]
            output_vector = outputs[i]
            self.weights += self.learning_rate * np.outer(input_vector, output_vector)

    def predict(self, input_vector):
        return np.dot(input_vector, self.weights)



# Define input and output patterns
inputs = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
outputs = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])

# Initialize the Hebbian network
network = HebbianNetwork(input_size=2, output_size=2)

# Train the network
network.train(inputs, outputs)

# Test the network
test_input = np.array([1, 0])
predicted_output = network.predict(test_input)

print(f"Predicted output for input {test_input}: {predicted_output}")