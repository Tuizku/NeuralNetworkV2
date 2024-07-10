import numpy as np
import os
from joblib import dump, load

from layer import Layer # other scripts use this
from neurons import Neuron, ReluNeuron, SigmoidNeuron, SoftmaxNeuron # other scripts use this
from funcs import *

def save_model(model):
    """
    Parameters:
    model (neuralnetwork.Model): This gets saved to the "models" directory by it's model name.
    """

    filepath = os.path.join("models", f"{model.name}.joblib")
    dump(model, filepath)

def load_model(name, dir = "models"):
    """
    Parameters:
    name (string): The name of the model WITHOUT EXTENSION.
    dir (string): The directory where the model is. Use os.path.join to create the string path
    """

    filepath = os.path.join(dir, f"{name}.joblib")
    loaded = load(filepath)
    return loaded



class Model:
    def __init__(self, name, layers):
        self.name = name
        self.layers = layers
        self.randomize()

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
    
    def randomize(self):
        for i in range(1, len(self.layers)):
            self.layers[i].randomize_params(self.layers[i - 1])

    def predict(self, inputs):
        self.layers[0].input(inputs)

        # Activate all the layers after input neurons
        for i in range(1, len(self.layers)):
            self.layers[i].activate(self.layers[i - 1])

    def predict_random(self):
        # Send the inputs
        inputs = np.random.uniform(0.0, 1.0, size=len(self.layers[0].neurons))
        self.layers[0].input(inputs)

        # Activate all the layers after input neurons
        for i in range(1, len(self.layers)):
            self.layers[i].activate(self.layers[i - 1])



    def set_training_data(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def train(self, epochs, learning_rate):
        """
        Uses Stochastic Gradient Descent. Doesn't work well with larger networks...
        If this project ever continues, implementing Mini-Batch Gradient Descent is essential.
        """

        print("train started")

        if not np.any(self.x_train) or not np.any(self.y_train) or not np.any(self.x_test) or not np.any(self.y_test):
            warning("training data error, data missing")
            return
        if len(self.x_train) != len(self.y_train):
            warning("training data error, list lengths are not the same")
            return
        
        epochs_per_print = max(1, np.ceil(epochs / 100))
        datas_per_print = 50
        train_len = len(self.x_train)

        # Train the training data as many times as epochs is
        for epoch_i in range(epochs):

            # Train a single data at a time
            for train_i in range(train_len):
                inputs = self.x_train[train_i]
                self.predict(inputs)

                # Calculate prediction errors (a - y)
                outputs = self.layers[-1].activations
                wanted_outputs = self.y_train[train_i]
                prediction_errors = np.zeros(len(outputs))
                for i in range(len(outputs)):
                    prediction_errors[i] = outputs[i] - wanted_outputs[i]
                
                # Calculate gradients
                error_terms = self.get_error_terms(prediction_errors)
                weight_gradients, bias_gradients = self.get_gradients(error_terms)
                self.apply_gradients(weight_gradients, bias_gradients, learning_rate)

                # Calculate current cost between training datas
                if train_i % datas_per_print == 0 and train_len >= 100:
                    print(f"[epoch={epoch_i + 1}, train_i={train_i}] cost: {self.get_cost()}")
            
            # Calculate current cost between epochs
            if epoch_i % epochs_per_print == 0 or epoch_i == epochs - 1:
                print(f"[epoch={epoch_i + 1}] cost: {self.get_cost()}")
        
    def get_cost(self):
        return np.mean((self.y_train[-1] - self.layers[-1].activations) ** 2)

    def get_error_terms(self, avg_pred_errs):
        layers_len = len(self.layers)
        
        error_terms = [] # not a numpy 2d array, because we don't want static dimensions. this 2d array can be turned into lists and those into numpy lists

        # Go through almost all layers backwards. X increases because the lists are easier to handle that way.
        for x in range(layers_len - 1):
            # Layer variables
            layer_i = layers_len - 1 - x
            layer = self.layers[layer_i]
            layer_neurons_len = len(layer.neurons)
            layer_err_terms = []
            prev_layer = self.layers[layer_i - 1]

            # Calculate output layer error terms
            if x == 0:
                # Go through all of the output neurons and calculate
                for y in range(layer_neurons_len):
                    neuron = layer.neurons[y]
                    pred_err = avg_pred_errs[y]
                    z = neuron.get_z(prev_layer, layer, y)
                    a = neuron.activation(z=z, previous_layer=prev_layer, layer=layer, neuron_id=y)

                    err_term = 2 * pred_err * neuron.derivative(a)
                    layer_err_terms.append(err_term)

            # Calculate earlier layer error terms
            else:
                next_layer = self.layers[layer_i + 1]
                transposed_weights =  next_layer.weights.transpose()
                next_error_terms = error_terms[x - 1]

                # 1. Multiply transposed_weights and next_error_terms (returns a list, where the index is the neuron's index)
                weighted_error_terms = np.matmul(transposed_weights, next_error_terms)
                
                # 2. Loop through all neurons and multiply the derivative sigmoid value with the weighted error terms
                for y in range(layer_neurons_len):
                    neuron = layer.neurons[y]
                    z = neuron.get_z(prev_layer, layer, y)
                    a = neuron.activation(z=z, previous_layer=prev_layer, layer=layer, neuron_id=y)

                    err_term = weighted_error_terms[y] * neuron.derivative(a)
                    layer_err_terms.append(err_term)

            error_terms.append(layer_err_terms)
        return error_terms

    def get_gradients(self, error_terms):
        bias_gradients = error_terms
        weight_gradients = []

        # Calculate the weight gradients with outer product
        for x in range(len(error_terms)):
            layer_i = len(self.layers) - 1 - x
            layer_err_terms = error_terms[x]
            last_activations = self.layers[layer_i - 1].activations
            w_layer_gradients = np.outer(layer_err_terms, last_activations)
            weight_gradients.append(w_layer_gradients)
                                    
        return weight_gradients, bias_gradients

    def apply_gradients(self, weight_gradients, bias_gradients, learning_rate):
        
        # Test if gradient lengths match
        gradient_layers_len = len(weight_gradients)
        if gradient_layers_len != len(bias_gradients):
            warning("weight and bias gradient lengths don't match")
            return
        
        # Apply backwards to all layers that have gradients
        for x in range(gradient_layers_len):
            layer_i = len(self.layers) - 1 - x
            layer = self.layers[layer_i]
            prev_layer = self.layers[layer_i - 1]

            for y in range(len(layer.neurons)):
                # Update weights
                for prev_y in range(len(prev_layer.neurons)):
                    old_weight = layer.weights[y][prev_y]
                    w_gradient = weight_gradients[x][y][prev_y]
                    new_weight = old_weight - (learning_rate * w_gradient)
                    layer.weights[y][prev_y] = new_weight
                
                # Update biases
                old_bias = layer.biases[y]
                b_gradient = bias_gradients[x][y]
                new_bias = old_bias - (learning_rate * b_gradient)
                layer.biases[y] = new_bias