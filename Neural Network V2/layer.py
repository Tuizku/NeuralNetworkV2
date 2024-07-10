import numpy as np
import math

from funcs import *
from neurons import *


# Settings
bias_randomize_limit = 0.1


# A single layer in a neural network.
class Layer:
    def __init__(self, type, length):
        self.type = type
        
        # Set weights and biases to none
        self.weights = None
        self.biases = None
        self.activations = None

        # Create the neurons for this layer
        self.neurons = []
        for i in range(length):
            new_neuron = type()
            self.neurons.append(new_neuron)
    
    # Randomizes the weights and biases
    def randomize_params(self, previous_layer):
        previous_layer_len = len(previous_layer.neurons)
        layer_len = len(self.neurons)

        # Creates the weight matrix (2D array) with random weights inside the limits
        # The first axis defines to which neuron is the weight connected in this layer
        # The second axis defines the neuron in the previous layer, where it is connected
        weight_limit = np.sqrt(6 / (previous_layer_len + layer_len))
        self.weights = np.random.uniform(-weight_limit, weight_limit, size=(layer_len, previous_layer_len))

        # Creates the bias vector with random biases inside the limits
        self.biases = np.random.uniform(-bias_randomize_limit, bias_randomize_limit, size=layer_len)
    
    # I don't think this is being used at this time
    def import_params(self, weights, biases):
        self.weights = weights
        self.biases = biases


    def activate(self, previous_layer):
        """
        Activates all the neurons in this layer. Used in predicting.
        """

        self.activations = np.zeros(len(self.neurons))
        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            a = neuron.activate(previous_layer, self, i)
            self.activations[i] = a
    
    def input(self, inputs):
        """
        Input the activations for this layer. Make sure that the inputs match the expected dimensions and etc.
        """

        if len(self.neurons) == len(inputs):
            self.activations = inputs
        else:
            self.activations = np.zeros(len(self.neurons))
            warning("Inputting layer's activations failed. The length of neurons wasn't equal to the length of inputs. | layer.py -> Layer -> input()")
        

