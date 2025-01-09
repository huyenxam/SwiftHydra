import argparse
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from models import Encoder, Decoder, VAEModel, MambaNet  
from sklearn.metrics import classification_report
import numpy as np
import warnings
warnings.filterwarnings('ignore')  
import random
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
import sklearn.metrics as skm
from tqdm import tqdm
from copy import deepcopy
from utils import *


parser = argparse.ArgumentParser(description='VAE Model Training Script with Dynamic Arguments')
parser.add_argument('--name_dataset', type=str, default='4_breastw', help='Name of the input dataset file')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for data loading')
parser.add_argument('--num_epochs_vae', type=int, default=1000, help='Number of epochs for training the VAE model')
parser.add_argument('--num_epochs_detector', type=int, default=600, help='Number of epochs for training the Detector model')
parser.add_argument('--learning_rate_z', type=float, default=1e-5, help='Learning rate for the VAE model')
parser.add_argument('--learning_rate_d', type=float, default=1e-3, help='Learning rate for the Detector model')
parser.add_argument('--num_episodes', type=int, default=50, help='Number of episodes')
parser.add_argument('--num_steps_per_episoder', type=int, default=400, help='Number of steps per episode')
parser.add_argument('--test_size', type=float, default=0.7, help='Proportion of the dataset to use for testing')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for entropy weighting')
parser.add_argument('--beta', type=float, default=1.55, help='Coefficient for adjusting the contribution of entropy in the loss function')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training (e.g., "cpu" or "cuda:0")')


# Parse the arguments
args = parser.parse_args()

# Use parsed arguments for settings
BATCH_SIZE = args.batch_size
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS_VAE = args.num_epochs_vae
NUM_EPOCHS_DETECTOR = args.num_epochs_detector
LR_Z = args.learning_rate_z
LR_D = args.learning_rate_d
NUM_EPISODES= args.num_episodes
NUM_GEN_DATA= args.num_steps_per_episoder
GAMMA = args.gamma
TEST_SIZE = args.test_size
BETA = args.beta



# Load dataset
dataset_path = "./Classical/" +  args.name_dataset + ".npz"
data = np.load(dataset_path, allow_pickle=True)

# Extract features (X) and labels (y) from the dataset
X = torch.tensor(data['X'])
y = torch.tensor(data['y']).float().unsqueeze(1)

# Standardize the feature data to have a mean of 0 and standard deviation of 1
scaler = StandardScaler()
X = torch.tensor(scaler.fit_transform(X)).float()

# Split the dataset into training and test sets (80% test, 20% train)
D_train, D_test, yD_train, yD_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=21)
Xtrain_1 = D_train[yD_train.squeeze() == 1]
Xtrain_0 = D_train[yD_train.squeeze() == 0]
ytrain_1, ytrain_0 = [1] * len(Xtrain_1), [0] * len(Xtrain_0)

# VAE traning
encoder = Encoder(input_dim = Xtrain_1.shape[1], hidden_dim = 256, latent_dim = 64)
decoder = Decoder(input_dim = 64, latent_dim = 256, hidden_dim = 256, output_dim=Xtrain_1.shape[1])
encoder.apply(init_weights)
decoder.apply(init_weights)
vae_model = VAEModel(encoder, decoder)
vae_model.to(device)
optimizer_vae = Adam([{'params': vae_model.parameters()}], lr=1e-5)
for param in vae_model.parameters():
    param.requires_grad = True
encoder = vae_model.Encoder
decoder = vae_model.Decoder


# Mamba traning
detector_model = MambaNet(Xtrain_1.shape[1])
detector_model.to(device)
optimizer_fc = Adam([{'params': detector_model.parameters()}], lr=5e-4)
criterion_fc = nn.BCELoss()

num_samples_1 = len(Xtrain_1)
num_samples_0 = len(Xtrain_0)

# Randomly sample from class 0 to match the number of samples in class 1
if num_samples_0 > num_samples_1:
    indices_0 = random.sample(range(num_samples_0), num_samples_1)
    Xtrain_0_balanced = Xtrain_0[indices_0]
    ytrain_0_balanced = [0] * num_samples_1
else:
    Xtrain_0_balanced = Xtrain_0
    ytrain_0_balanced = ytrain_0

# Combine the balanced datasets for training
Xtrain_balanced = torch.cat([Xtrain_1, Xtrain_0_balanced], dim=0)
ytrain_balanced = torch.tensor(ytrain_1 + ytrain_0_balanced).float().unsqueeze(1)


# Chuẩn bị DataLoader cho dữ liệu cân bằng
train_dataset  = TensorDataset(Xtrain_balanced, ytrain_balanced)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset  = TensorDataset(D_test, yD_test)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=True)




for param in detector_model.parameters():
        param.requires_grad = True

num_epochs_FC = 300
for epoch in range(num_epochs_FC):
    epoch_loss = 0
    for batch_idx, data_pair in enumerate(train_loader):
        x = data_pair[0].float().to(device)
        y = data_pair[1].float().to(device)
        
        outputs = detector_model(x).squeeze()
        loss = criterion_fc(outputs, y.squeeze(1))

        # In thông tin cần theo dõi
        print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.2f}, Outputs = {outputs[:5].cpu().detach().numpy()}, Labels = {y[:5].cpu().numpy()}")

        optimizer_fc.zero_grad()
        loss.backward()
        optimizer_fc.step()

        epoch_loss += loss.item()



evaluate_model(test_loader, detector_model, device)




for param in vae_model.parameters():
    param.requires_grad = False
for param in detector_model.parameters():
    param.requires_grad = False
encoder = vae_model.Encoder
decoder = vae_model.Decoder


for i in range(1):
    data_iter = iter(train_loader)
    data = next(data_iter)[0].to(device)
    random_index = random.randint(0, data.size(0) - 1)
    mean, log_var, z_hat = encoder(data[random_index].unsqueeze(0).to(device))

    z_hat = z_hat.detach()
    z_hat.requires_grad = True
    z_optimizer = Adam([z_hat], lr=1e-2)
    y_true = torch.zeros(1).to(device)
    
    tmp_x = None
    tmp_y = None
    for j in range(5000):
        variance = np.exp(log_var.cpu().numpy())
        std_dev = np.sqrt(variance)
        lower_bound = mean.cpu() - 3 * std_dev
        upper_bound = mean.cpu() + 3 * std_dev

        is_in_range = ((z_hat.cpu() >= lower_bound) & (z_hat.cpu() <= upper_bound)).all().item()
        if not is_in_range:
            print(f"Step {i}, Iter {j}: z_hat out of bounds, stopping.")
            break

        x_hat = decoder(z_hat)
        y_hat = detector_model(x_hat)

        # In thông tin cần theo dõi
        print(f"Step {i}, Iter {j}: Loss = {loss.item():.5f}, z_hat = {z_hat.detach().cpu().numpy()}, Mean = {mean.detach().cpu().numpy()}, Log Var = {log_var.detach().cpu().numpy()}")

        loss = F.mse_loss(y_hat, y_true)
        loss.backward()
        z_optimizer.step()


