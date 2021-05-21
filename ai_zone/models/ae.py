import torch
from ai_zone.config import Config

class Autoencoder(torch.nn.Module):
    def __init__(self, c:Config):
        super(Autoencoder, self).__init__()
        self.e = torch.nn.LSTM(input_size=c.get("features"), hidden_size=64, num_layers=2, dropout=0.1)
        self.bridge = torch.nn.LSTM(input_size=c.get("features"), hidden_size=64, num_layers=2, dropout=0.1)