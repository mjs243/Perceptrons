# Multilayer Perceptron, multiple training methods

import numpy as np

np.set_printoptions(precision=8)


class MLP:
    def __init__(self, input_size, layer_sizes, initial_weights, initial_biases):
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.weights = [np.array(layer_weights) for layer_weights in initial_weights]
        self.biases = [np.array(layer_biases) for layer_biases in initial_biases]
        self.activations = [np.zeros(n) for n in layer_sizes]
        self.little_E = np.zeros(self.layer_sizes[-1])

    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)

    def feed_forward(self, inputs):
        activations = inputs
        for i, weight in enumerate(self.weights):
            net_inputs = np.dot(weight, activations) + self.biases[i]
            activations = self.activation(net_inputs)
            self.activations[i] = activations
        print(f"\nFeedforward Activations: {self.activations[-1]}")
        return self.activations[-1]

    def backprop2(self, inputs, target, eta):
        self.feed_forward(inputs)
        self.little_E = target - self.activations[-1]
        output_deltas = self.little_E * self.derivative(self.activations[-1])
        deltas = [output_deltas]
        for i in reversed(range(len(self.weights) - 1)):
            layer = i + 1
            errors = np.dot(self.weights[layer].T, deltas[0])
            delta = errors * self.derivative(self.activations[layer - 1])
            deltas.insert(0, delta)

        for i in range(len(self.weights)):
            layer = i + 1
            inputs_to_use = self.activations[i - 1] if i > 0 else inputs
            self.weights[i] += eta * np.outer(deltas[i], inputs_to_use)
            self.biases[i] += eta * deltas[i]

        big_E = 0.5 * np.sum(self.little_E**2)
        return big_E, deltas

    # training method 1
    """ For each cycle of this training procedure, present the first input/output pair, 
        perform the back propagation technique to update the weights, then present the second 
        input/output pair and again perform the back propagation technique to update the 
        weights. This constitutes a single cycle. Perform 15 such cycles and determine the errors 
        E for each input/output pair. This method essentially updates the weights by alternately 
        presenting each input/output pair … the first pair, and then the second pair and so on. 
        After the 15th training cycle, present the input values to the network and print out the 
        total Error (Big E) and the final weights associated with each input/output pair """

    def train1(self, inputs, targets, eta, epochs):
        print("Training Method 1\n")
        inputs = np.array(inputs)
        targets = np.array(targets)

        for epoch in range(epochs):
            print(f"--- Epoch {epoch + 1} ---")
            total_big_E = 0

            for input_index, (input, target) in enumerate(zip(inputs, targets)):
                print(
                    f"\nTraining on Input #{input_index + 1}: {input} with Target: {target}"
                )
                self.feed_forward(input)
                print(
                    f"Activations for Input {input_index + 1}: {self.activations[-1]}"
                )
                big_E, _ = self.backprop2(input, target, eta)
                print(
                    f"Little e after training on Input {input_index + 1}: {self.little_E}"
                )
                print(f"Big E after training on Input {input_index + 1}: {big_E}")
                total_big_E += big_E

            print(f"\nEnd of Epoch {epoch + 1}, Total Big E: {total_big_E}")
            print("Updated Weights and Biases:")
            for i, w in enumerate(self.weights):
                print(f"Layer {i + 1} weights: {w}")
            for i, b in enumerate(self.biases):
                print(f"Layer {i + 1} biases: {b}")
            print("\n")

    # training method 2
    """ In this method, we update weights for one input/output pair for 15 iterations
        of the FFBP algorithm, then present the second input/output pair and run the FFBP 
        algorithm to update weights for another 15 iterations. Thus, for each cycle, present the 
        first input/output pair, run the FFBP for 15 iterations, then present the second 
        input/output pair and run the FFBP for another 15 iterations. This second set of iterations 
        therefore begins updating the weights from the values obtained after the first 15 iterations
        with the first input/output pair. After the training of the second input/output pair, present 
        the input of the first pair, print out the total Error (Big E) and the final weights associated 
        with it, then present the input of the second pair and print out the total Error (Big E). """

    def train2(self, inputs, targets, eta, epochs):
        print("Training Method 2")
        inputs = np.array(inputs)
        targets = np.array(targets)

        for input_index, (input, target) in enumerate(zip(inputs, targets)):
            print(
                f"\nTraining on Input #{input_index + 1}: {input} for {epochs} epochs"
            )

            for epoch in range(epochs):
                print(f"\nEpoch {epoch + 1}/{epochs} for Input {input_index + 1}")
                self.feed_forward(input)
                print(
                    f"Activations for Input {input_index + 1}: {self.activations[-1]}"
                )
                big_E, _ = self.backprop2(input, target, eta)
                print(f"Little e after Epoch {epoch + 1}: {self.little_E}")
                print(f"Big E after Epoch {epoch + 1}: {big_E}")
                print("Updated Weights:")
                for i, w in enumerate(self.weights):
                    print(f"Layer {i + 1}: {w}")
                print("Updated Biases:")
                for i, b in enumerate(self.biases):
                    print(f"Layer {i + 1}: {b}")


# implementation

input_size = 2
layer_sizes = [2, 1]
initial_weights = [[[0.3, 0.3], [0.3, 0.3]], [[0.8, 0.8]]]

initial_biases = [[0.0, 0.0], [0.0]]

mlp = MLP(input_size, layer_sizes, initial_weights, initial_biases)

inputs = [[1, 1], [-1, -1]]
targets = [[0.9], [0.05]]
eta = 1.0
epochs = 15

mlp.train2(inputs, targets, eta, epochs)

# getting values for assignment

est = mlp.feed_forward([1.0, 1.0])
print(est)
error = 0.9 - est
print(f"lil e: {error}")

big_E = 0.5 * (error**2)
print(f"big e: {big_E}")
