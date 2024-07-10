import math
import numpy as np

# Parent neuron
class Neuron:

    # Calculates the sum of weights multiplied by activations + bias (called "z")
    def get_z(self, previous_layer, layer, neuron_id):
        z = layer.biases[neuron_id]
        for i in range(len(previous_layer.neurons)):
            z += layer.weights[neuron_id][i] * previous_layer.activations[i]
        return z


# A child neuron with a sigmoid activation function
class ReluNeuron(Neuron):
    def activate(self, previous_layer, layer, neuron_id):
        """Activates this neuron and returns the activation value"""

        z = Neuron.get_z(self, previous_layer, layer, neuron_id)
        a = self.activation(z=z) # relu activation function
        return a

    def activation(self, z = None, previous_layer = None, layer = None, neuron_id = None):
        """
        This is the unique activation function of this neuron.
        
        Expects these variables:
        - z
        """
        
        return max(0, z)

    def derivative(self, a):
        return np.where(a > 0, 1, 0)


# A child neuron with a sigmoid activation function
class SigmoidNeuron(Neuron):

    def activate(self, previous_layer, layer, neuron_id):
        """Activates this neuron and returns the activation value"""

        z = Neuron.get_z(self, previous_layer, layer, neuron_id)
        a = self.activation(z=z) # sigmoid activation function
        return a
    
    
    def activation(self, z = None, previous_layer = None, layer = None, neuron_id = None):
        """
        This is the unique activation function of this neuron.
        
        Expects these variables:
        - z
        """

        return 1 / (1 + math.exp(-z))

    def derivative(self, a):
        return a * (1 - a)


# A child neuron with a softmax activation function
class SoftmaxNeuron(Neuron):
    
    def activate(self, previous_layer, layer, neuron_id):
        """Activates this neuron and returns the activation value"""

        return self.activation(previous_layer=previous_layer, layer=layer, neuron_id=neuron_id)

    def activation(self, z = None, previous_layer = None, layer = None, neuron_id = None):
        """
        This is the unique activation function of this neuron.
        
        Expects these variables:
        - previous_layer
        - layer
        - neuron_id
        """

        exp_zs = np.zeros(len(layer.neurons))
        for y in range(len(layer.neurons)):
            exp_zs[y] = np.exp(layer.neurons[y].get_z(previous_layer, layer, y))
        
        numerator = exp_zs[neuron_id]
        denominator = np.sum(exp_zs)
        return numerator / denominator

    def derivative(self, a):
        return a * (1 - a)