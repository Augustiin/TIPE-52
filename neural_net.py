import numpy as np
import matplotlib.pyplot as plt
import time
import os

class Dense_Layer:

    def __init__(self, n_inputs, n_neurons):
        self.shape = (n_inputs, n_neurons)
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # gradient des paramètres
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # gradient des valeurs
        self.dinputs = np.dot(dvalues, self.weights.T)

class Layer_Input:

    def forward(self, inputs):
        self.output = inputs

class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(inputs, 0)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Tanh:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.tanh(inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs *= 1 - np.power(self.output, 2)

class Activation_Linear:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

class Activation_SoftMax:

    def forward(self, inputs):
            numerator = np.exp(inputs - np.max(inputs, axis=1,
            keepdims=True)) # pour éviter l'overflow
            denominator = np.sum(numerator, axis=1, keepdims=True)

            self.output = numerator / denominator

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:

    def calculate(self, output, y):

        sample_loss = self.forward(output, y)

        loss = np.mean(sample_loss)

        return loss

class Loss_MeanSquaredError(Loss):

    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        # Number of outputs in every sample
        outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples

class Loss_CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        # Nombre de classes
        samples = len(dvalues)

        # Number of labels in every sample
        labels = len(dvalues[0])

        # Clip data to prevent division by 0
        dvalues_clipped = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # If labels are sparse, convert to one-hot
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues_clipped
        self.dinputs = self.dinputs / samples

class Optimizer_SGD: # SGD

    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
        self.iterations = 0

    def pre_update_params(self):
        pass

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

    def post_update_params(self):
        self.iterations += 1

class Model:

    def __init__(self):
        self.input_layer = Layer_Input()
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, type=''): # type = classification / regression
        self.loss = loss
        self.optimizer = optimizer
        self.type = type

    def forward(self, X):
        self.input_layer.forward(X)

        current_layer = self.input_layer
        current_output = X
        for layer in self.layers:
            layer.forward(current_output)
            current_layer = layer
            current_output = layer.output

        return current_layer.output # à la dernière itération c'est la prédiction

    def backward(self, output, y):

        self.loss.backward(output, y)

        current_layer = self.loss
        current_output = output
        for layer in reversed(self.layers):
            layer.backward(current_layer.dinputs)
            current_layer = layer
            current_output = output

    def optimize(self):
        self.optimizer.pre_update_params()

        for layer in self.layers:
            if type(layer).__name__ == 'Dense_Layer':
                self.optimizer.update_params(layer)

        self.optimizer.post_update_params()

    def save_params(self, X, y, predictions):
        # save params
        structure_size = []
        structure_activation = []
        learning_rate = self.optimizer.__dict__['learning_rate']
        loss_name = self.loss.__class__.__name__
        optimizer_name = self.optimizer.__class__.__name__

        for layer in self.layers:
            if type(layer).__name__ == 'Dense_Layer':
                structure_size.append(layer.__dict__['shape'])

            elif 'Activation' in type(layer).__name__:
                structure_activation.append(layer.__class__.__name__[11:])

        return structure_size, structure_activation, learning_rate, loss_name, optimizer_name

    def train(self, X, y, *, epochs=1, train=True):

        self.epoch = epochs
        epoch_training = []
        data_training = []
        accuracies = []

        for i in range(1, epochs + 1):
            # forward pass
            self.output = self.forward(X)

            # calculate loss and accuracy
            loss = self.loss.calculate(self.output, y)
            # calculate accuracy for classification
            accuracy = 0
            if self.type == 'classification':

                if len(y.shape) == 2:
                    y = np.argmax(y, axis=1)

                true = 0
                for (a,b) in zip(np.argmax(self.output, axis=1), y):
                    if a == b:
                        true += 1

                accuracy = true/len(y)

            if train :
                # bacwkard pass
                self.backward(self.output, y)

                # optimization
                self.optimize()

            # data for summary & plot
            if i == 1:
                data_training.append(self.output)
                epoch_training.append(i)
                print(f'epoch: {i}, loss: {loss:.3f}, accuracy: {accuracy:.3f}, lr: {self.optimizer.learning_rate}')
                if self.type == 'classification':
                    accuracies.append(accuracy)

            elif not i % (self.epoch//6):
                print(f'epoch: {i}, loss: {loss:.3f}, accuracy: {accuracy:.3f}, lr: {self.optimizer.learning_rate}')
                data_training.append(self.output)
                epoch_training.append(i)
                if self.type == 'classification':
                    accuracies.append(accuracy)

            elif i == self.epoch:
                print(f'epoch: {i}, loss: {loss:.3f}, accuracy: {accuracy:.3f}, lr: {self.optimizer.learning_rate}')
                data_training.append(self.output)
                epoch_training.append(i)
                if self.type == 'classification':
                    accuracies.append(accuracy)


        return self.save_params(X, y, self.output), data_training, epoch_training, accuracies
