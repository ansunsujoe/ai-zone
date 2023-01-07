from torch import nn

class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)
    
    def forward(self, x):
        return self.linear(x)