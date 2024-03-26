from Layer import Layer
import numpy as np

class DenseLayer(Layer):
    ## n_Inputs reffers to the dimensions and n_Ouputs is the neurons number
    def __init__(self, n_Inputs, n_Outputs, act_function) -> None:
        super().__init__()
        self.weights = np.random.rand(n_Outputs, n_Inputs) * 2 - 1
        self.bias =  np.random.rand(n_Outputs, 1)          * 2 - 1
        self.activation_function =  act_function # Activation Function has two params: the function and his derivate
        
        self.inputs = None
        self.z = None


    def forward_prop(self, inputs) -> np.ndarray:
        self.inputs = inputs
        self.z = np.dot(self.weights, self.inputs) + self.bias
        outputs = self.activation_function[0](self.z)


        return outputs
    
    def gradient_descent(self, cost, learnig_rate) -> None:

        self.weights -= learnig_rate * np.dot(cost, np.sum(self.inputs.T, 0, keepdims= True)) 
        self.bias -= learnig_rate * cost


