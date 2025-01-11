import argparse
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from models import Encoder, Decoder, VAEModel, MambaNet  
import numpy as np
import warnings
warnings.filterwarnings('ignore')  
import random
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
from copy import deepcopy
from utils import *
import logging

# Cấu hình logging
logging.basicConfig(
    filename="training_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.info("Training process started.")

parser = argparse.ArgumentParser(description='VAE Model Training Script with Dynamic Arguments')
parser.add_argument('--name_dataset', type=str, default='1_ALOI', help='Name of the input dataset file')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for data loading')
parser.add_argument('--num_epochs_vae', type=int, default=1000, help='Number of epochs for training the VAE model')
parser.add_argument('--num_epochs_detector', type=int, default=600, help='Number of epochs for training the Detector model')
parser.add_argument('--learning_rate_z', type=float, default=1e-3, help='Learning rate for the VAE model')
parser.add_argument('--learning_rate_d', type=float, default=1e-4, help='Learning rate for the Detector model')
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


# VAE Model
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


# Mamba Model
detector_model = MambaNet(Xtrain_1.shape[1])
detector_model.to(device)
optimizer_fc = Adam([{'params': detector_model.parameters()}], lr=5e-4)
criterion_fc = nn.BCELoss()


for i in range(100):
    ytrain_1, ytrain_0 = [1] * len(Xtrain_1), [0] * len(Xtrain_0)

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

    # Combine the balanced datasets (both classes) for training
    Xtrain_balanced = torch.cat([Xtrain_1, Xtrain_0_balanced], dim=0) 
    ytrain_balanced = torch.tensor(ytrain_1 + ytrain_0_balanced).float().unsqueeze(1).to(device) 

    # Prepare DataLoader for the VAE training dataset (positive class)
    ytrain_1 = torch.tensor(ytrain_1).float().unsqueeze(1).to(device) 
    vae_dataset = TensorDataset(Xtrain_1, ytrain_1)  
    vae_loader = DataLoader(vae_dataset, batch_size=BATCH_SIZE, shuffle=True)  

    # Prepare DataLoader for the negative class dataset
    ytrain_0 = torch.tensor(ytrain_0).float().unsqueeze(1).to(device)  
    sample_dataset = TensorDataset(Xtrain_0, ytrain_0)  
    sample_loader = DataLoader(sample_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Prepare DataLoader for the combined detector dataset (balanced classes)
    detector_dataset = TensorDataset(Xtrain_balanced, ytrain_balanced) 
    detector_loader = DataLoader(detector_dataset, batch_size=BATCH_SIZE, shuffle=True) 

    # Prepare DataLoader for testing dataset
    test_dataset = TensorDataset(D_test, yD_test)  
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True) 


    ##################### VAE Model #########################
    for param in vae_model.parameters():
        param.requires_grad = True
    encoder = vae_model.Encoder
    decoder = vae_model.Decoder

    # Initialize a list to store latent space representations (z-values)
    z_density = []
    
    # Training loop for VAE
    for epoch in range(NUM_EPOCHS_VAE):
        z_density = []
        
        for batch_idx, data_pair in enumerate(vae_loader):
            x = data_pair[0].float().to(device)
            optimizer_vae.zero_grad()
            x_hat, mean, log_var, z = vae_model(x)
            
            z_density.extend(z.cpu().detach().numpy().tolist())
            total_loss, re_loss, KL_loss = loss_vae(x, x_hat, mean, log_var)     
            total = total_loss.sum() / BATCH_SIZE  
            re = re_loss.sum() / BATCH_SIZE 
             
            logging.info(f"Epoch {epoch}, Batch {batch_idx}: Total Loss = {total.item():.2f}, "
                         f"Reproduction Loss = {re.item():.2f}, KL Divergence = {KL_loss.mean().item():.2f}, "
                         f"Latent Mean = {mean.mean().item():.2f}")

            total.backward()
            optimizer_vae.step()
            

    for param in vae_model.parameters():
        param.requires_grad = False
    for param in detector_model.parameters():
        param.requires_grad = False
    encoder = vae_model.Encoder
    decoder = vae_model.Decoder
    
    
    ##################### Detector Model #############################
    
    for param in detector_model.parameters():
            param.requires_grad = True

    for epoch in range(NUM_EPOCHS_DETECTOR):
        epoch_loss = 0
        for batch_idx, data_pair in enumerate(detector_loader):
            x = data_pair[0].float().to(device)
            y = data_pair[1].float().to(device)
            
            outputs = detector_model(x).squeeze()
            loss = criterion_fc(outputs, y.squeeze(1))

            logging.info(f"Epoch {epoch}, Batch {batch_idx}: Total Loss = {loss.item():.5f}, ")

            optimizer_fc.zero_grad()
            loss.backward()
            optimizer_fc.step()
            epoch_loss += loss.item()

    evaluate_model(test_loader, detector_model, device)
    
    for param in detector_model.parameters():
        param.requires_grad = False


    ##################### Explore ######################
    
    for sample_idx in range(NUM_GEN_DATA):
        data_iter = iter(vae_loader)
        batch_data = next(data_iter)[0].to(device)
        random_sample_idx = random.randint(0, batch_data.size(0) - 1)

        mean_latent, log_var_latent, z_sample = encoder(batch_data[random_sample_idx].unsqueeze(0).to(device))

        z_sample = z_sample.detach()
        z_sample.requires_grad = True
        z_optimizer = Adam([z_sample], lr=5e-4)
        target_label = torch.zeros(1).to(device)  

        updated_latent = deepcopy(z_sample)
        x_hat = None
        for iteration in range(NUM_EPISODES):
            temp_density_data = deepcopy(z_density)
            variance = np.exp(log_var_latent.cpu().detach().numpy())
            std_dev = np.sqrt(variance)
            lower_bound = mean_latent.cpu().detach().numpy() - 3 * std_dev
            upper_bound = mean_latent.cpu().detach().numpy() + 3 * std_dev

            lower_bound_tensor = torch.tensor(lower_bound, device=z_sample.device)
            upper_bound_tensor = torch.tensor(upper_bound, device=z_sample.device)
            is_in_valid_range = ((z_sample.cpu() >= lower_bound_tensor.cpu()) & (z_sample.cpu() <= upper_bound_tensor.cpu())).all().item()

            if not is_in_valid_range:
                logging.warning(f"Sample {sample_idx}, Iter {iteration}: z_sample out of bounds, stopping.")
                break

            temp_density_data.extend(z_sample.cpu().detach().numpy().tolist())
            updated_latent = deepcopy(z_sample)

            kde_estimator = KernelDensity(kernel='gaussian', bandwidth=0.5)
            kde_estimator.fit(np.array(temp_density_data))
            latent_samples = np.array(temp_density_data)
            log_density_values = kde_estimator.score_samples(latent_samples)
            density_values = np.exp(log_density_values)
            entropy_score = entropy(density_values)
             
            reconstructed_sample = decoder(z_sample)            
            predicted_label = detector_model(reconstructed_sample)
            if predicted_label.item() >  0.2:
                x_hat = reconstructed_sample
                break
            
            loss_value = F.mse_loss(predicted_label, target_label) 
            logging.info(f"Step {sample_idx}, Iter {iteration}: Loss = {loss_value.item():.5f}, z_sample = {z_sample.detach().cpu().numpy()}")
                        
            loss_value.backward()
            z_optimizer.step()

        if x_hat is not None and x_hat.squeeze(0).shape[0] == Xtrain_1.shape[1]:
            x_hat = x_hat.clone().detach().requires_grad_(True).squeeze(0).unsqueeze(0).to(Xtrain_1.device) 
            Xtrain_1 = torch.cat((Xtrain_1, x_hat), dim=0)  

            new_label = torch.tensor([[1]]).float().to(ytrain_1.device)  
            ytrain_1 = torch.cat((ytrain_1, new_label), dim=0).clone().detach().requires_grad_(True) 


        z_density.extend(updated_latent.cpu().detach().numpy().tolist())

