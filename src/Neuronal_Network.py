import numpy as np

class Neuronal_Network():
    def __init__(self,  cost_funct) -> None:
        self.cost_fuction = cost_funct
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def train(self, x_train, y_train, learning_rate = 0.1):
        for i in range(1):
            ## Excute the forward prop
            Y_hat = self.__forward_propagation(x_train).T

            print(Y_hat.shape)
            print(y_train.shape)

            ## Calculate the cost
            cost = self.cost_fuction[0](Y_hat, y_train)
            print("The cost is {}".format(cost))




    def __forward_propagation(self, X) -> np.ndarray:
        output = X

        for layer in self.layers:
            output = layer.forward_prop(output)
            

        return output
    
    def __backward_propagation(self, a, Y) -> np.ndarray:
        grad = []

        for layer in reversed(self.layers):

            z =  layer.Z
            cost = None

            if(layer == self.layers[len(self.layers) - 1]):
                print("Entre aca")

                cost = self.cost_fuction[1](a, Y) * layer.activation_function[1](z)
                print(z)
                grad.append(cost)
            else: 
                z = layer.Z
                cost = np.dot(cost, layer.weights) * layer.activation_function[1](z)
                grad.append(cost)

        return grad
            
        


