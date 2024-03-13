import numpy as np

class Layer:

    def __init__(self) -> None:
        self.inputs =  None
        self.outputs = None
        self.activation_f = None
    

    def forward_propagation() -> np.ndarray: 
        raise NotImplemented
    
    def backward_propagation() -> np.ndarray:
        raise NotImplemented