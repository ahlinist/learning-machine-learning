import numpy as np


class Perceptron:
    def __init__(self, inputs, bias=1.0):
        self.weights = (np.random.rand(inputs + 1) * 2) - 1
        self.bias = bias

    def run(self, x):
        x_sum = np.dot(np.append(x, self.bias), self.weights)
        return self.sigmoid(x_sum)

    def set_weights(self, weights):
        self.weights = np.array(weights, dtype=object)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class NeuralNetwork:
    def __init__(self, layers, eta=0.5):
        self.layers = layers
        self.network = []
        self.values = []
        self.eta = eta
        self.d = []

        for layer in range(len(self.layers)):
            self.values.append([])
            self.network.append([])
            self.d.append([])
            self.values[layer] = np.zeros(self.layers[layer], dtype=object)
            self.d[layer] = np.zeros(self.layers[layer], dtype=object)

            for neuron in range(self.layers[layer]):
                if layer == 0:
                    inputs_number = self.layers[layer]
                else:
                    inputs_number = self.layers[layer] + 1  # +1 for bias
                self.network[layer].append(Perceptron(inputs_number))

    def set_weights(self, weights):
        for layer in range(len(self.layers)):
            for neuron in range(self.layers[layer]):
                self.network[layer][neuron].set_weights(weights[layer][neuron])

    def print_weights(self):
        print()
        for i in range(len(self.layers)):
            for j in range(self.layers[i]):
                print('Layer: ', i + 1, ", Neuron: ", j + 1, ", Weight: ", self.network[i][j].weights)
        print()

    def run(self, x):
        for i in range(len(self.network)):
            for j in range(len(self.network[i])):
                if i == 0:
                    self.values[i][j] = self.network[i][j].run(x)
                else:
                    self.values[i][j] = self.network[i][j].run(self.values[i - 1])
        return self.values[-1]

    def propagate_back(self, x, y):
        x = np.array(x, dtype=object)
        y = np.array(y, dtype=object)
        output = self.run(x)
        error = y - output
        mse = sum(error ** 2) / self.layers[-1]

        self.d[-1] = output * (1 - output) * error

        for layer_index in reversed(range(len(self.network) - 1)):
            for neuron_index in range(len(self.network[layer_index])):
                fwd_error = 0.0
                neuron_output = self.values[layer_index][neuron_index]
                for next_layer_neuron_index in range(len(self.network[layer_index + 1])):
                    neuron_out_weight = self.network[layer_index + 1][next_layer_neuron_index].weights[neuron_index]
                    next_layer_neuron_error = self.d[layer_index + 1][next_layer_neuron_index]
                    fwd_error += neuron_out_weight * next_layer_neuron_error
                    self.d[layer_index][neuron_index] = neuron_output * (1 - neuron_output) * fwd_error

        for layer_index in range(len(self.network)):
            for neuron_index in range(len(self.network[layer_index])):
                for weight_index in range(len(self.network[layer_index][neuron_index].weights)):
                    if layer_index == 0:
                        correction = self.eta * self.d[layer_index][neuron_index] * np.append(x, 1)[weight_index]
                        self.network[layer_index][neuron_index].weights[weight_index] += correction
                    else:
                        error = self.d[layer_index][neuron_index]
                        input_value = np.append(self.values[layer_index - 1], 1)[weight_index]
                        correction = self.eta * error * input_value
                    self.network[layer_index][neuron_index].weights[weight_index] += correction
        return mse
