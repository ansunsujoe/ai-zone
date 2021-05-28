from torch.autograd.grad_mode import F
import torch
from torch import nn
from ai_zone.config import Config

class Encoder(nn.Module):
    def __init__(self, c:Config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=c.get("features"),
            hidden_size=c.get("latent_dim"),
            num_layers=2,
            dropout=0
        )
        
    def forward(self, X):
        X, (hidden_state, cell_state) = self.lstm(X)
        # Take the last hidden state (this is where we go from timeseries to n-dimensional vector)
        last_hidden_state = hidden_state[-1, :, :]
        return last_hidden_state
    
class Decoder(nn.Module):
    def __init__(self, c:Config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=c.get("features"),
            hidden_size=c.get("latent_dim"),
            num_layers=2,
            dropout=0
        )
        self.fc = nn.Linear(
            hidden_size=c.get("latent_dim"), 
            output_size=c.get("features")
        )
        self.timesteps = c.get("timesteps")
    
    def forward(self, X):
        X = X.unsqueeze(1).repeat(1, self.timesteps, 1)
        X, (hidden_state, cell_state) = self.lstm(X)
        X = X.reshape((-1, self.timesteps, self.hidden_size))
        out = self.fc(X)
        return out
    
class Autoencoder(nn.Module):
    def __init__(self, c:Config):
        super().__init__()
        self.encoder = Encoder(c)
        self.decoder = Decoder(c)
        
    def forward(self, X):
        encoder_output = self.encoder(X)
        decoder_output = self.decoder(encoder_output)
        return encoder_output, decoder_output
    
    def fit(self, X):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        criterion = nn.MSELoss(reduction='mean')
    
    def encode(self, X):
        self.eval()
        return self.encoder(X)
    
    def decode(self, X):
        self.eval()
        return self.decoder(X).squeeze()
    
    def load(self, model_fp):
        self.is_fitted = True
        self.load_state_dict(torch.load(model_fp))