import os
import numpy as np
import neuralnetwork as nn
import debug
from joblib import load

class mnist_data:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test



# Create the test1 neural network and open it in the debug window
def test1():
    layers = [nn.Layer(nn.Neuron, 4), 
              nn.Layer(nn.SigmoidNeuron, 6),
              nn.Layer(nn.SigmoidNeuron, 2)]
    
    # Test training data, if positive input is in top 2 neurons, output is top neuron, 
    # if positive input is in bottom 2 neurons, output is bottom neuron
    x_train = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    y_train = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

    model = nn.Model("test1", layers)
    model.set_training_data(x_train, y_train, x_train, y_train)
    print("opening debug\n")
    debug.open(model)

    print("finished\n")

# Test creating a neural network and open it in the debug window
def custom():
    # Setup input layer
    input_neurons = int(input("input neuron count: "))
    layers = [nn.Layer(nn.Neuron, input_neurons)]

    # Setup other layers
    neuron_types = [nn.SigmoidNeuron, nn.SoftmaxNeuron]
    for i in range(2, 6):
        print(f"\nLAYER {i}")
        neuron_count = int(input("layer neuron count: "))
        type_num = int(input("layer neuron type (0 = sigmoid, 1 = softmax): "))
        layer_type = neuron_types[type_num]
        layers.append(nn.Layer(layer_type, neuron_count))

        if input("stop adding layers (blank = no, yes = stop): ") == "yes":
            break

    # Create and open model
    model = nn.Model("custom", layers)
    print("opening debug\n")
    debug.open(model)

    print("finished\n")

# Create a neural network for number recognition and open it in the debug window.
# Note: My neural network can't get the cost down, and it's probably because i use Stochastic Gradient Descent.
#       Minibatches would be a lot better way to train a model, with a lot less noise in training.
def open_number_recognition():
    layers = [nn.Layer(nn.Neuron, 784),
              nn.Layer(nn.ReluNeuron, 128),
              nn.Layer(nn.ReluNeuron, 128),
              nn.Layer(nn.SoftmaxNeuron, 10)]

    path = os.path.join("data", "mnist.joblib")
    mnist_data = load(path)

    x_train = np.array(mnist_data.x_train[:1000]).reshape(mnist_data.x_train[:1000].shape[0], -1)
    y_train = np.eye(10)[mnist_data.y_train[:1000]]
    x_test = np.array(mnist_data.x_test[:1000]).reshape(mnist_data.x_test[:1000].shape[0], -1)
    y_test = np.eye(10)[mnist_data.y_test[:1000]]

    model = nn.Model("number_recognition", layers)
    model.set_training_data(x_train, y_train, x_test, y_test)
    print("opening debug\n")
    debug.open(model)

    print("finished\n")



while True:
    command = input("command: ")
    if command == "help":
        print("""
default:
- exit

create a neural network and open it in the debug window:
- test1 (with training data)
- custom (no training data)

create a more complex neural network and open it in the debug window (my training doesn't really work at this scale anymore)
- open_number_recognition / onr
            """)
    elif command == "exit": break
    elif command == "test1": test1()
    elif command == "custom": custom()
    elif command == "open_number_recognition" or command == "onr": open_number_recognition()
    