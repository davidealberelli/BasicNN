from numpy import random
from numpy import dot
from numpy import array
import numpy as np
import pdb
import matplotlib.pyplot as plt

debug = True
output_list = []
error_list = []

class network():
    def __init__(self):
        random.seed(1)
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    #Neural net sigmoid functions    
    def sigmoid(self, x):
      return 1 /(1+(np.exp(-x)))

    def adjust(self, x):
        return x * (1 - x)

    def process(self, inputs):
        new_x = dot(inputs, self.synaptic_weights)
        return self.sigmoid(new_x)

    def train(self, In, Out, iterations):
        for iteration in range(iterations):
            output = self.process(In)
            error = Out - output
            adjustment = dot(In.T, error * self.adjust(output))            
            self.synaptic_weights += adjustment

            if(debug 
            and iteration % 10 == 0):
                print(f'Iteration = {iteration}')
                print(f'Output = {output.T}')
                print(f'Error = {error.T}')
                print(f'Adjustment = {adjustment.T}')
            
            print(f'Weights = {self.synaptic_weights.T}')
            output_list.append(output)
            error_list.append(error)

    def process_test(self, inputs):
        new_x = dot(inputs, self.synaptic_weights)
        #pdb.set_trace()
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
cheating_classify.train(training_labels_cheating_in, training_labels_cheating_out, 100)

def predict(scores):
  return(cheating_classify.process_test(array(scores)))

print(f'Not cheating, should be close to 0: {predict([10,8,6])}')
print(f'Cheating, should be close to 1: {predict([8,9,10])}')

#pdb.set_trace()
output_vec = np.array(output_list)
x = range(len(output_vec))
for i in range(len(output_vec[0])):
    plt.scatter(x,[pt[i] for pt in output_vec],label = 'id %s'%i)
plt.ylabel('output')
plt.show()