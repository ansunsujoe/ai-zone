import numpy as np
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn
    
    def forward(self, x):
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray, n_epochs: int, batch_size: int, 
            learning_rate: float, validation_split: float = 0.0):
        pass