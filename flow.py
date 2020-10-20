import numpy as np
import torch
import torch.nn as nn


class RealNVP(nn.Module):
    """
    Normalizing flow based on "Density estimation using Real NVP"
    (L.Dinh, J.Sohl-Dickstein, S.Bengio - 2017) and inspired from
    the architecture of senya-ashukha/real-nvp-pytorch on github.
    This structure allows to constrain the mapping with a waveform.
    """

    def __init__(self, s_net, t_net, mask, prior):
        # s_net (scale) and t_net (translation) could be MLP or CNN
        super(RealNVP, self).__init__()
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = nn.ModuleList([t_net() for _ in range(len(mask))])
        self.s = nn.ModuleList([s_net() for _ in range(len(mask))])

    def g_forward(self, z, waveform):
        # Training
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            x_wf = torch.cat([x_, waveform], dim=1)
            s = self.s[i](x_wf) * (1 - self.mask[i])
            t = self.t[i](x_wf) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f_inverse(self, x, waveform):
        # Sampling
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            z_wf = torch.cat([z_, waveform], dim=1)
            s = self.s[i](z_wf) * (1 - self.mask[i])
            t = self.t[i](z_wf) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x, waveform):
        # Natural log probability after the change of variable
        z, logp = self.f_inverse(x, waveform)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize, waveform):
        # Forward mapping to sample the posterior
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.g_forward(z.view(batchSize, -1), waveform)
        return x


# Scaling and translation networks models for the coupling layer :
# (Fully connected networks: 4 layers with Leaky ReLU activation)

def s_net(n_in=9198+4, n_out=4):
    """
    Sequential pytorch model for scale network.
    """
    return nn.Sequential(nn.Linear(n_in, 16),
                         nn.LeakyReLU(),
                         nn.Linear(16, 16),
                         nn.LeakyReLU(),
                         nn.Linear(16, n_out),
                         nn.Tanh())


def t_net(n_in=9198+4, n_out=4):
    """
    Sequential pytorch model for translation network.
    """
    return nn.Sequential(nn.Linear(n_in, 16),
                         nn.LeakyReLU(),
                         nn.Linear(16, 16),
                         nn.LeakyReLU(),
                         nn.Linear(16, n_out))


def set_4d_masks():
    """
    Masks for a 4 parameters inference (4 times).
    """
    masks_4d = torch.from_numpy(
        np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0],
                  [1, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1],
                  [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0],
                  [1, 1, 0, 0], [0, 1, 1, 1], [1, 0, 1, 1],
                  [1, 1, 0, 1], [1, 1, 1, 0]]*4).astype(np.float32))
    return masks_4d
