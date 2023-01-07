from torch import nn

class MLP(nn.Module):
    def __init__(self, layer_sizes: list[int], normalize_batches: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)
    
    def forward(self, x):
        return self.linear(x)