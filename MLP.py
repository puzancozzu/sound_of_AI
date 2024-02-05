### buidling the own muti layer percepton - NEURAL NETWORK

import numpy as np 


### defining a class :Multi layer perceptorn(MLP)
class MLP:

    #### defining the constructor of the class with different arguments
    ### default NN: input = 3 , 2 hidden layer with 3 and 5 neurons, and outputs=2
    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        ### number of hidden layer is a list and values defines the number of neuron in each layer
        

        ### internal representation of layers 
        ### get a list -where each itemm in a list represent the number of neuron in the layer 
        ### and layers moves from  0 index to number of layer we have

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        ### initiate a random weight for layers

        ### each layer we have weight matrix - w
        ### create a array/matrix of different dimenisions - with random values between 0 and 1
        ### here 2d array - rows: current layer , colum : number of neurons in subsequent layer
        self.weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)


    ### defines how inputs travels through diff layers
    def froward_propagate(self, inputs):

        activations = inputs

        ### take inputs values
        ### move from left to right layers
        ### in each layer we perform ceratin calculation : net inputs and activations
        for w in self.weights:
            ### calcualte the net inputs for first layer

            ## PERFORMS  the matrix multiplication between input and weights
            net_inputs = np.dot(activations, w)

            ###calculate the activations
            activations = self._sigmoid(net_inputs)

        return activations
    

    def _sigmoid(self, x):
        #### it is just a sigmoid activation function - we know the formula of it
        return 1/(1 + np.exp(-x))

if __name__ == "__main__":

    ### create an MLP

    ### we can change in number of inputs, outputs and hidden layer 
    mlp = MLP()


    ### create some inputs 
    inputs = np.random.rand(mlp.num_inputs)


    ### perform forward prop

    outputs = mlp.froward_propagate(inputs)


    ### print results
    print("The network input is : {}". format(inputs))
    print("The network output is : {}".format(outputs))
