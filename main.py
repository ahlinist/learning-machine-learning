from perceptron import Perceptron, NeuralNetwork


def main():
    network = NeuralNetwork(layers=[2, 1])

    for i in range(3000):
        mse = 0.0
        mse += network.propagate_back([0, 0], [0])
        mse += network.propagate_back([1, 0], [1])
        mse += network.propagate_back([0, 1], [1])
        mse += network.propagate_back([1, 1], [0])
        mse /= 4

        if i % 100 == 0:
            print(mse)

    network.print_weights()

    print(network.d)

    print('XOR gate trained:')
    print('0 0 = {0:.10f}'.format(network.run([0, 0])[0]))
    print('0 1 = {0:.10f}'.format(network.run([0, 1])[0]))
    print('1 0 = {0:.10f}'.format(network.run([1, 0])[0]))
    print('1 1 = {0:.10f}'.format(network.run([1, 1])[0]))



if __name__ == '__main__':
    main()
