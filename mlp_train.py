### multi layer perceptron and training the network

import numpy as np 
from random import random
'''
#######
1 . save the activations and derivatives  - for back propagations
2 . implement back propagation
3 . implement gradient descent
4 . implement train method 
5 . train out nets with some dummy dataset
6 . make some predictions
'''


### defining a class :Multi layer perceptorn(MLP)
class MLP:

    #### constructor for MLP
    ''' takes the numbers of inputs - num_inputs (int)
        varaiable of hidden layers - hidden layers (list) : list of ints for hidden layers
        numbers of outputs - num_outputs (int)
    '''
    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):
        self.num_inputs = num_inputs
        self.num_hidden = hidden_layers
        self.num_outputs = num_outputs

        ### create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        ### create random connection weights for the layers
        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

        ### create a activations for each layer
        ### create a dummy activation for each layer - array of zeros = num of layers we have in NN
        ### we will have a list of array where each  array in th list reperesenation a activation for a given layer
        ### store a list in instance variable
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations


        ### for derivatives of layers : derivatives of ERROR function w.r.t to weights
        ### 2D array : rows: num of neurons in a layer , columns: num of neurons in subsequnet layer 
        derivatives = []
        for i in range(len(layers)-1):    ### eg: for three layers NN we have only 2 weight matices
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives



    ### forward propagation
    def froward_propagate(self, inputs):
        '''
        computes forward propagation of the net based onn input signals.
        input signal : inputs (ndarray)
        '''
        ### input layer activation is just a input itself i.e. no avtivation function for input layer
        activations = inputs

        ## save the activation for back propagatoion
        self.activations[0] = activations


        ### iterate through net layers
        for i, w in enumerate(self.weights):
            
            ## calc matrix mult btwn previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            ## apply sigmoid activations
            activations = self._sigmoid(net_inputs)
            ## save activations
            self.activations[i+1] = activations

            '''
            why in [i+1] not in 'i'?--- a_3 = s(h_3) ,  h_3 = a_2 * w_2
            activation of 3rd layer is sigmoid function of H_3 and H_3 is equal to matix mul of wight2 and a_2
            so for i = 2, weight matrix is also for second one and activation connected with 2nd wight matrix is
            activation for 3rd layer (2+1)
            '''
        return activations
    

    ### define back propagation function
    ### back propagate error from output layer to input layer
    ### agr: error (ndaary) the error to back propoagate
    ### returns: erro(ndaary) the final error of the input
    def back_propagate(self, error, verbose=False):
        ## we want to iterate from right layer of net to left - back propagation , so reveresed

        ''' we are using sigmoid activation s(h) and quadratic error function - error 
            we have , dE/dW_i = (y - a_[i+1]) s'(h_[i+1]) a_i 
            y - a = error 
            s'(gh_[i+1]) = s(h_[i+1]) (1- s(h_[i+1]))
            s(h_[i+1]) = a_[i+1]

            #### one step back derivtaives i.e of i-1's layer
            dE/dW_[i-1] = (y - a_[i+1]) s'(h_[i+1]) W_i s'(h_i) a_[i-1]
        '''
        for i in reversed(range(len(self.derivatives))):
            
            ##get activation for previous layer
            activations = self.activations[i+1]

            ## apply sigmoid derivative fnction
            delta = error * self._sigmoid_derivatives(activations)

            ### for delta  : ndarray([0.1, 0.2]) ----> ndarray([[0.1, 0.2]]) 2-d array with single row
            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            ## get current activation of layer
            current_activations = self.activations[i]  

            ### rearrange it so it is vertiacal vector - column matrix : ndarry([0.1, 0.2]) --> ndarray([[0.1], [0.2]])
            current_activations= current_activations.reshape(current_activations.shape[0], -1)

            ### here, for derivatives -- matrix mult we need to arrange {delta , current-activations} such that they can be multiplied
            self.derivatives[i] = np.dot(current_activations, delta_reshaped)

            ### for next layer, i -1 th layer 
            ### here, error = (y - a_[i+1]) s'(h_[i+1]) W_i 
            ### delta = = (y - a_[i+1]) s'(h_[i+1])
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivatives for W{} : {}".format(i, self.derivatives[i]))
        

    ### creating Gradient Descent
    def  gradient_descent(self, learning_rate=1):

        ## loop through all the weights
        for i in range(len(self.weights)):

            ## retriving the weights and its coresponding derivativies
            weights = self.weights[i]
            # print("original Weights{}: {}".format(i, weights))
            deravitives = self.derivatives[i]


            ## update weights
            weights += deravitives * learning_rate
            # print("Updated Weights{}: {}".format(i, weights))


    ### create training methods for training NN
    def train(self, inputs, targets, epochs, learning_rate):

        ## Epoch - how many times iteratively the entier data is feed to NN
        for i in range(epochs):
            sum_error = 0 

            ## packing and unpacking, (inputs and targets)
            for j, input in enumerate(inputs):
                target = targets[j]

                ## forward propagation
                output = self.froward_propagate(input)

                ## calculate error
                error = target - output

                ## back propagation
                self.back_propagate(error)

                ## applying gradient descent
                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output)

            ## report error after each Epoch
            print(f"Error : {sum_error / len(inputs)} at epoch {i+1}")

        print("Training Complete!")
        print("====================")


    ### defining mean squared error : 1/2(target - output)^2
    def _mse(self, target, output):
        return np.average((target - output)**2)

    ### define deravitives of sigmoid functions
    def _sigmoid_derivatives(self, x):
        return x * (1.0-x)

    def _sigmoid(self, x):
        #### it is just a sigmoid activation function - we know the formula of it
        return 1.0/(1 + np.exp(-x))






if __name__ == "__main__":

    ## create an MLP
    ## inputs : 2 values, hidden layer : 1 with 5 perceptrons , output : 1 
    mlp = MLP(2, [5], 1)

    ## create a dataset to train a net for sum operation
    input = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])      ## array([[0.1,0.2], [0.3,0.4]])
    target = np.array([[i[0] + i[1]] for i in input])   ## array([[0.3], [0.7]])

    # print(len(input))

    # print(len(target))

    # train our MLP
    mlp.train(input, target, 100, 0.1)


    ## create dummpy data(inputs and targets)
    input = np.array([0.1, 0.3])
    target = np.array([0.4])
    # ## trying to do sum -- expecting network to learn how to do sum

    output = mlp.froward_propagate(input)
    print("\n\n\n")
    print(f"Our network output for {input[0]} + {input[1]} is : {output[0]}")

    error = (target[0] - output[0]) * 100/ target[0]
    print("Error in prediction is : {} %".format(error))
