import torch
import sklearn.metrics as skm
import numpy as np
from sklearn.metrics import classification_report
import torch.nn.functional as F
import torch.nn as nn


def evaluate_model(test_loader, model, device, mode='test'):
    y_true, y_pred, scores = [], [], []

    model.eval()
    with torch.no_grad():
        for data_pair in test_loader:
            x = data_pair[0].float().to(device)
            y = data_pair[1].float().to(device)
            
            y_true.append(y[0].item() if mode != 'test' else y.item())
            outputs = model(x)
            score = outputs[0].item() if mode != 'test' else outputs.item()
            scores.append(score)
            
            y_pred.append(1 if score > 0.2 else 0)

    auc_roc = skm.roc_auc_score(np.array(y_true), np.array(scores))
    report = classification_report(y_true, y_pred)

    print(f"AUC-ROC: {auc_roc}\n{report}")
    return auc_roc


def loss_vae(x, x_hat, mean, log_var):
    total_loss = 0
    reconstruction_loss = F.mse_loss(x_hat, x, reduction='sum') 
    KL = torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    total_loss = reconstruction_loss + 0.001 * KL 
    
    return total_loss, reconstruction_loss, KL


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

