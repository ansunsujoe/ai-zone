from torch import nn
from ai_zone.model_architecture.heads import ModelHead
from ai_zone.model_architecture.activations import get_activation_from_string
from ai_zone.models.neural_network import NeuralNetwork

class MLP(NeuralNetwork):
    def __init__(self, input_size: int, layer_sizes: list[int], head: ModelHead, 
                 activation: str = "relu", normalize_batches: bool = False):
        super().__init__()
        self.head = head
        self.normalize_batches = normalize_batches
        self.activation = get_activation_from_string(activation)
        
        # Store layers
        self.linear_layers = []
        self.batch_norm_layers = []
        
        # First layer set
        self.linear_layers.append(nn.Linear(in_features=input_size, out_features=layer_sizes[0]))
        if normalize_batches:
            self.batch_norm_layers.append(nn.BatchNorm1d(num_features=layer_sizes[0]))
        
        # Subsequent layer sets
        for i in range(len(layer_sizes) - 1):
            self.linear_layers.append(nn.Linear(in_features=layer_sizes[i], out_features=layer_sizes[i+1]))
            self.batch_norm_layers.append(nn.BatchNorm1d(num_features=layer_sizes[1]))
    
    def forward(self, x):
        # First forward block
        out = self.linear_layers[0](x)
        if self.normalize_batches:
            out = self.batch_norm_layers[0](out)
        out = self.activation(out)
        
        # Subsequent forward blocks
        for i in range(1, len(self.linear_layers)):
            out = self.linear_layers[i](out)
            if self.normalize_batches:
                out = self.batch_norm_layers[i](out)
            out = self.activation(out)
            
        # The model head will be the output function
        out = self.head(out)
        return out