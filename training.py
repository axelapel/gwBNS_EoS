import corner
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import flow
import hdf5datasets


################################# Extraction ##################################


# Local
local_path_in = "data/trainset_freq_projected_nonoise.hdf"
local_path_out = "out/"
local_path_model_checkpoints = "checkpoints/"

# Cluster
cluster_path_in = "/scratch/alapel/data/trainset_freq_projected_nonoise.hdf"
cluster_path_out = "/scratch/alapel/out/"
cluster_path_model_checkpoints = "/scratch/alapel/checkpoints/"

# Datasets
batch_size = 1000
trainset = hdf5datasets.HDF5EoSDataset(local_path_in)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

# GPU if available
if torch.cuda.is_available():
    dev = "cuda:0"
    dtype = torch.cuda.FloatTensor
else:
    dev = "cpu"
    dtype = torch.FloatTensor

print("Device = {}".format(dev))


############################## Training Real NVP ##############################

# Coupling layer
s_net = flow.s_net
t_net = flow.t_net

# Masks
masks = flow.set_4d_masks()

# Initial distribution
normal = torch.distributions.MultivariateNormal(torch.zeros(4), torch.eye(4))

# Instance of the flow
rNVP = flow.RealNVP(s_net, t_net, masks, normal)
rNVP.to(torch.device(dev))

# Optimizer
optimizer = torch.optim.Adam(
    [p for p in rNVP.parameters() if p.requires_grad == True], lr=1e-4)

# Training
max_epoch = 50
epochs = np.arange(max_epoch)
for epoch in epochs:
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

    if epoch + 1 == 25 or epoch + 1 == 50:
        print("Model saved.")
        torch.save({
            "epoch": epoch,
            "model_state_dict": rNVP.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }, cluster_path_model_checkpoints + "trained_rNVP_{}.pth".format(epoch+1))

    print("[epoch {}/{}] loss = {:.3f}".format(epoch+1, max_epoch, loss))
print("Training over.")

################################### Testing ###################################

test_waveform = trainset[0][0]
params = [trainset[0][1], trainset[0][2], trainset[0][3], trainset[0][4]]

# The number of samples have to be a fraction of the batch_size
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
samples[:, 0] *= 1.7
samples[:, 1] *= 1.36
samples[:, 2] *= 1000
samples[:, 3] *= 1000

# TODO : improve the plot
figure = corner.corner(
    samples, labels=[r"$m_1$", r"$m_2$", r"$\Lambda_1$", r"$\Lambda_2$"],
    show_titles=True, truths=[params[0]*1.7, params[1]*1.36,
                              params[2]*1000, params[3]*1000])
plt.savefig("figures/posterior_masses_lambdas.png")
