import torch
import torch.nn as nn
import math
from mamba_ssm import Mamba

class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input1 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_input3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear(hidden_dim, latent_dim)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.training = True
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)  # sampling epsilon        
        z = mean + var * epsilon  # reparameterization trick
        return z
    
    def forward(self, x):
        h_ = self.relu(self.FC_input(x))
        h_ = self.relu(self.FC_input1(h_))
        h_ = self.relu(self.FC_input2(h_))
        h_ = self.relu(self.FC_input3(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        return mean, log_var, z


class Decoder(nn.Module):
    
    def __init__(self, input_dim, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        
        self.FC_input = nn.Linear(input_dim, latent_dim)
        self.FC_hidden_1 = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden_2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_hidden_3 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
        
    def forward(self, z):
        h = self.relu(self.FC_input(z))
        h = self.relu(self.FC_hidden_1(h))
        h = self.relu(self.FC_hidden_2(h))
        h = self.relu(self.FC_hidden_3(h))
        # x_hat = torch.sigmoid(self.FC_output(h))
        x_hat = self.FC_output(h)
        
        return x_hat
    

class VAEModel(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAEModel, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
                
    def forward(self, x):
        mean, log_var, z = self.Encoder(x)
        x_hat = self.Decoder(z)
        
        return x_hat, mean, log_var, z


class FullyConnectedNet(nn.Module):
    def __init__(self, input_size):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(1024, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x



import torch
import torch.nn as nn
import math
from mamba_ssm import Mamba


class MambaNet(nn.Module):
    def __init__(self, input_size):
        super(MambaNet, self).__init__()
        
        self.mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=input_size, # Model dimension d_model
            d_state=128,  # SSM state expansion factor
            d_conv=2,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(0.2)
        # self.fc2 = nn.Linear(256, 256)
        # self.dropout2 = nn.Dropout(0.2)
        # self.fc3 = nn.Linear(256, 256)
        # self.dropout3 = nn.Dropout(0.2)
        # self.fc4 = nn.Linear(256, 256)
        # self.dropout4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(256, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.mamba(x.unsqueeze(1)).squeeze(1)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        # x = self.leaky_relu(self.fc2(x))
        # x = self.dropout2(x)
        # x = self.leaky_relu(self.fc3(x))
        # x = self.dropout3(x)
        # x = self.leaky_relu(self.fc4(x))
        # x = self.dropout4(x)
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x