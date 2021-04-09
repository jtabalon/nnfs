import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

layer_outputs = np.dot(weights, inputs) + biases

print(type(layer_outputs), layer_outputs)

# >>> (<type 'numpy.ndarray'>, array([ 4.8  ,  1.21 ,  2.385])) 

####################################################

a = [1, 2, 3]
np.array([a])
#np.array([[1, 2, 3]])

#OR

np.expand_dims(np.array(a), axis=0)

#####################################################

a = [1, 2, 3]
b = [2, 3, 4]

a = np.array([a])
b = np.array([b]).T

print(np.dot(a, b))
#>>> [[20]]

######################################################

inputs = [[1.0, 2.0, 3.0, 2.5],
         [2.0 ,5.0, -1.0, 2.0],
         [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

outputs = np.dot(inputs, np.array(weights).T) + biases
print(outputs)
