import numpy as np

class Neuronal_Network():
    def __init__(self,  cost_funct, metrics) -> None:
        self.cost_fuction = cost_funct
        self.metrics = metrics
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def fit(self, x_train, y_train, epochs = 50, learning_rate = 0.1, validation_data = None):
        for epoch in range(epochs):
            ## Execute forward prop
            Y_hat = self.__forward_propagation(x_train).T

            ## Calculate the cost
            loss = self.cost_fuction[0](Y_hat, y_train)
            acc = self.metrics[0](Y_hat, y_train)

            val_Y_hat = self.__forward_prop(validation_data[0]).T
            val_loss = self.cost_fuction[0](val_Y_hat, validation_data[1])
            val_acc = self.metrics[0](val_Y_hat, validation_data[1])

            print("Epoch {}/{} -> loss: {} - acc: {} - val_loss: {} - val_acc: {}"
                  .format(epoch+1, epochs, loss, acc, val_loss, val_acc))

            ## Execute backward prop
            error_matrix = self.__backward_propagation(Y_hat, y_train)

            ## Execute gradient descent
            self.__gradient_descent(error_matrix, learning_rate)
            




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

                grad.insert(0, error)

            else:
                error = np.dot(np.dot(W_l.T, error), np.sum(layer.activation_function[1](Z).T, 0, keepdims = True))
                error = np.sum(error, 1, keepdims=True)

                W_l = layer.weights
                grad.insert(0, error)

        return grad
    
    
    def __gradient_descent(self, error, learning_rate):
        for i, layer in enumerate(reversed(self.layers)):
                error_num = len(error) - i - 1 ## This is to obtain the index grad corresponding to the current layer
                
                layer.gradient_descent(error[error_num], learning_rate)




    def __forward_prop(self, X) -> np.ndarray:
        output = X

        for layer in self.layers:
            output = layer.forward_propagation(output)
            

        return output
            
        


