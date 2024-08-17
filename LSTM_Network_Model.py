import torch
import torch.nn as nn
import torch.optim as optim

# Parameters
input_layer = 600
output_layer = 3
hidden_layer = 256
num_layers = 1


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(hidden_dim, output_dim, num_layers, batch_first=True)
    
    def forward(self, x):
        # Initialize hidden state and cell state for first LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Pass through the first LSTM layer
        output1, _ = self.lstm1(x, (h0, c0))
        
        # Initialize hidden state and cell state for second LSTM
        h1 = torch.zeros(self.num_layers, x.size(0), self.output_dim).to(x.device)
        c1 = torch.zeros(self.num_layers, x.size(0), self.output_dim).to(x.device)

        # Pass output of the first LSTM as input to the second LSTM
        output2, _ = self.lstm2(output1, (h1, c1))
        
        return output2

def load_model():
    model = LSTMModel(input_layer, hidden_layer, output_layer, num_layers)
    return model