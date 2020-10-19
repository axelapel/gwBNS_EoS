import corner
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import flow
import hdf5datasets


# GPU if available
if torch.cuda.is_available():
    dev = "cuda:0"
    dtype = torch.cuda.FloatTensor
else:
    dev = "cpu"
    dtype = torch.FloatTensor

print("Device = {}".format(dev))

# Local
local_path_in = "data/"
local_path_model_checkpoints = "checkpoints/"
local_path_fig = "figures/"

# Cluster
cluster_path_in = "/scratch/alapel/data/"
cluster_path_model_checkpoints = "/scratch/alapel/checkpoints/"
cluster_path_fig = "/scratch/alapel/figures/"

# Training file
last_epoch = 5
train_file = "trained_rNVP_{}.pth".format(last_epoch)

# Test waveforms
testset = hdf5datasets.HDF5EoSDataset(
    cluster_path_in + "testset_freq_projected_nonoise.hdf")
testloader = DataLoader(testset, batch_size=1, shuffle=True)

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

# Load training
checkpoint = torch.load(local_path_model_checkpoints + train_file)
rNVP.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# Input
test_waveform = testset[0][0]
params = [testset[0][1], testset[0][2], testset[0][3], testset[0][4]]

# The number of samples have to be a fraction of the batch_size
batch_size = 500
test_waveform_repeat = torch.from_numpy(
    np.tile(test_waveform, (batch_size, 1)))

# Sampling
samples = rNVP.sample(
    batch_size, test_waveform_repeat.type(dtype)).detach().cpu().numpy()
N = 1000
for i in range(N):
    x = rNVP.sample(batch_size, test_waveform_repeat.type(
        dtype)).detach().numpy()
    samples = np.vstack([samples, x])

# Renormalization of the sample
upper_values = [1.7, 1.36, 600]
samples[:, 0] *= upper_values[0]
samples[:, 1] *= upper_values[1]
samples[:, 2] *= upper_values[2]
samples[:, 3] *= upper_values[2]

# TODO : improve the plot
figure = corner.corner(
    samples, labels=[r"$m_1$", r"$m_2$", r"$\Lambda_1$", r"$\Lambda_2$"],
    show_titles=True, truths=[params[0]*upper_values[0], params[1]*upper_values[1],
                              params[2]*upper_values[2], params[3]*upper_values[2]])
# plt.savefig("figures/posterior_masses_lambdas.png")
plt.savefig(cluster_path_fig + "posterior_masses_lambdas.png")
