import numpy as np

class Neuronal_Network():
    def __init__(self,  cost_funct) -> None:
        self.cost_fuction = cost_funct
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def train(self, x_train, y_train, learning_rate = 0.1):
        for i in range(1):
            ## Execute the forward prop
            Y_hat = self.__forward_propagation(x_train).T

            ## Calculate the cost
            cost = self.cost_fuction[0](Y_hat, y_train)
            print("The cost is {}".format(cost))

            ## Excute the backward prop
            grad = self.__backward_propagation(Y_hat, y_train)




    def __forward_propagation(self, X) -> np.ndarray:
        output = X

        for layer in self.layers:
            output = layer.forward_prop(output)
            

        return output
    
    def __backward_propagation(self, Y_hat, Y) -> np.ndarray:
        grad = []
        error = [()]
        W = [()]

        for layer in reversed(self.layers):
            Z = layer.z ## The weighted sum of the layer

            if(layer == self.layers[len(self.layers) - 1]):
                W_l = layer.weights

                error = np.dot(layer.activation_function[1](Z), self.cost_fuction[1](Y_hat, Y))

                grad.append(error)

            else:
                error = np.dot(np.dot(W_l.T, error), np.sum(layer.activation_function[1](Z).T, 0, keepdims = True))
                error = np.sum(error, 1, keepdims=True)

                W_l = layer.weights
                grad.append(error)

        return grad
            
        


