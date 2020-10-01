from numpy import random
from numpy import dot
from numpy import array
import numpy as np
import pdb
import matplotlib.pyplot as plt

debug = False
output_list = []
error_list = []

np.set_printoptions(suppress=True)

class network():
    def __init__(self):
        random.seed(17)
        self.synaptic_weights = 2 * random.random((3, 1)) - 1
        self.synaptic_bias = 1

    #Neural net sigmoid cost function 
    def sigmoid(self, x):
        return 1 /(1+(np.exp(-x)))

    #Forward propagation for sigmoid cost function
    #we do a gradient descent, this is the "derivative of the sigmoid"
    #meaning that for the sigmoid f'(x) = f(x)*(1-f(x))
    def adjust(self, x):
        return x * (1 - x)

    def train(self, In, Out, iterations):
        for iteration in range(iterations):
            #Forward propagation
            output_layer_input =  dot(In, self.synaptic_weights) + self.synaptic_bias
            output = self.sigmoid(output_layer_input)
            
            #Backward propagation: this is where we adapt the parameters
            error = Out - output
            slope = self.adjust(output)
            
            #this is the back-propagated error
            output_delta = error * slope

            #we need to update the weights by this much
            weights_adjustment = dot(In.T, output_delta)    

            print(f'output = {output.T}')
            print(f'Iteration = {iteration}')
            print(f'Errors = {error.T}')
            print(f'Weights = {self.synaptic_weights.T}')

            self.synaptic_weights += weights_adjustment
            self.synaptic_bias += sum(error * self.adjust(output))
            
            output_list.append(output)
            error_list.append(error)

    def test(self, inputs):
        new_x = dot(inputs, self.synaptic_weights)# + self.synaptic_bias
        return self.sigmoid(new_x)

cheating_classify = network()
training_labels_cheating_in = array(
  [
    [10, 10, 10],
    [8, 9, 10],
    [10, 7, 6],
    [10, 9, 10],
    [9, 3, 2],
    [7, 6, 5]
  ])
training_labels_cheating_out = array([[1, 1, 0, 1, 0, 0]]).T
# Training:
# cheating_classify.train(training_labels_cheating_in, training_labels_cheating_out, 100)
# print(f'Not cheating, should be close to 0: {cheating_classify.test(array([10,8,6]))}')
# print(f'Cheating, should be close to 1: {cheating_classify.test(array([8,9,10]))}')

# output_vec = np.array(output_list)
# x = range(len(output_vec))
# for i in range(len(output_vec[0])):
#     plt.scatter(x,[pt[i] for pt in output_vec],label = 'id %s'%i)
# plt.ylabel('output')
# plt.show()

# error_vec = np.array(error_list)
# x = range(len(error_vec))
# for i in range(len(error_vec[0])):
#     plt.scatter(x,[pt[i] for pt in error_vec],label = 'id %s'%i)
# plt.ylabel('output')
# plt.show()

# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(0), random_state=1)
# input_train = [[1,2,3], [3,2,1]]
# output_train = [0, 1]
# clf.fit(input_train,output_train)

# print((clf.predict([[9,8,6],[6,8,9]])))

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(), alpha = 0)
clf.fit(training_labels_cheating_in, np.ravel(training_labels_cheating_out))
to_predict = [[10,8,6], [4,5,6], [7,6,6], [3,2,1]]
predicted = clf.predict(to_predict)
print(predicted)
print(clf.score(to_predict, [0,1,0,0]))
print(clf.loss_)
print(clf.n_iter_)
print(clf.coefs_)
print(clf.n_layers_)
