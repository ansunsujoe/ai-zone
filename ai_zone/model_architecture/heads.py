from torch import nn

class ModelHead:
    def __init__(self):
        pass

class ClassificationHead(ModelHead, nn.Module):
    def __init__(self, n_inputs: int, n_classes: int):
        super().__init__()
        
        # Create layers
        self.linear = nn.Linear(in_features=n_inputs, out_features=n_classes)
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        out = self.linear(x)
        out = self.softmax(out)
        return out
    

class LinearHead(ModelHead, nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int):
        super().__init__()
        
        # Create layers
        self.linear = nn.Linear(in_features=n_inputs, out_features=n_outputs)
        
    def forward(self, x):
        return self.linear(x)