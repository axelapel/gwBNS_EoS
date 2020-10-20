import corner
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import flow
import hdf5datasets


################################# Extraction ##################################

# Local
local_path_in = "data/"
local_path_model_checkpoints = "checkpoints/"
local_path_fig = "figures/"

# Cluster
cluster_path_in = "/scratch/alapel/data/"
cluster_path_model_checkpoints = "/scratch/alapel/checkpoints/"
cluster_path_fig = "/scratch/alapel/figures/"

# DATASETS
batch_size = 500
# Train
trainset = hdf5datasets.merge_sets(cluster_path_in + "train/")
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
# Validation
evalset = hdf5datasets.merge_sets(cluster_path_in + "evaluation/")
evaloader = DataLoader(evalset, batch_size=batch_size, shuffle=True)
# Test
testset = hdf5datasets.HDF5EoSDataset(
    cluster_path_in + "test/test_GW170817.hdf")
testloader = DataLoader(testset, batch_size=1, shuffle=True)

# GPU if available
if torch.cuda.is_available():
    dev = "cuda:0"
    dtype = torch.cuda.FloatTensor
else:
    dev = "cpu"
    dtype = torch.FloatTensor

print("Device = {}".format(dev))


############################## Training Real NVP ##############################

# Start from a previous checkpoint
from_file = False
epoch = 10
train_file = cluster_path_model_checkpoints + \
    "trained_rNVP_{}.pth".format(epoch)

# Coupling layer
s_net = flow.s_net
t_net = flow.t_net

# Masks
masks = flow.set_4d_masks()

# Initial distribution
normal_gaussian = torch.distributions.MultivariateNormal(
    torch.zeros(4), torch.eye(4))

# Instance of the flow
rNVP = flow.RealNVP(s_net, t_net, masks, normal_gaussian)
rNVP.to(torch.device(dev))

# Optimizer
optimizer = torch.optim.Adam(
    [p for p in rNVP.parameters() if p.requires_grad == True], lr=5e-4)

# Training
if from_file == True:
    checkpoint = torch.load(local_path_model_checkpoints + train_file)
    rNVP.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

max_epoch = 100
epochs = np.arange(max_epoch)
train_losses = np.zeros(max_epoch)
val_losses = np.zeros(max_epoch)
for epoch in epochs:
    rNVP.train()
    for i, (waveform, mass_1, mass_2, lambda_1, lambda_2) in enumerate(trainloader):
        # Float to tensor
        waveform = waveform.type(torch.FloatTensor)
        mass_1 = mass_1.type(torch.FloatTensor)
        mass_2 = mass_2.type(torch.FloatTensor)
        lambda_1 = lambda_1.type(torch.FloatTensor)
        lambda_2 = lambda_2.type(torch.FloatTensor)

        input = torch.cat([mass_1.view(-1, 1), mass_2.view(-1, 1),
                           lambda_1.view(-1, 1), lambda_2.view(-1, 1)], dim=1)

        loss = - rNVP.log_prob(input, waveform).mean()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    train_losses[epoch] = loss

    # Validation
    rNVP.eval()
    for i, (waveform, mass_1, mass_2, lambda_1, lambda_2) in enumerate(evaloader):
        # Float to tensor
        waveform = waveform.type(torch.FloatTensor)
        mass_1 = mass_1.type(torch.FloatTensor)
        mass_2 = mass_2.type(torch.FloatTensor)
        lambda_1 = lambda_1.type(torch.FloatTensor)
        lambda_2 = lambda_2.type(torch.FloatTensor)

        input = torch.cat([mass_1.view(-1, 1), mass_2.view(-1, 1),
                           lambda_1.view(-1, 1), lambda_2.view(-1, 1)], dim=1)

        loss_val = - rNVP.log_prob(input, waveform).mean()
    val_losses[epoch] = loss_val

    if epoch + 1 % 10 == 0:
        print("Model saved.")
        torch.save({
            "epoch": epoch,
            "model_state_dict": rNVP.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }, cluster_path_model_checkpoints + "trained_rNVP_{}.pth".format(epoch+1))

    print("[epoch {}/{}] loss = {:.3f}".format(epoch+1, max_epoch, loss))
print("Training over.")

fig, ax = plt.subplots()
ax.plot(epochs, train_losses, c="k", label="Training")
ax.plot(epochs, val_losses, c="tab:green", label="Validation")
ax.set(xlabel="Epochs", ylabel="Loss")
ax.grid()
ax.legend()
# plt.savefig("figures/loss.png")
plt.savefig(cluster_path_fig + "loss.png", transparent=True)
