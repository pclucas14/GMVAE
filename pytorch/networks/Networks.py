"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Gaussian Mixture Variational Autoencoder Networks

"""
import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from networks.Layers import *
from collections import OrderedDict as OD

# From Somewhere else
class Encoder(nn.Module):
    def __init__(self, channels, output_size):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            Residual(channels),
            Residual(channels, ds=True),
            nn.Conv2d(channels, output_size, 1)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, channels, input_size):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(input_size, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            Residual(channels, us=True),
            Residual(channels),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, 1, 1) #3 * 256, 1)
        )

    def forward(self, x):
        x = self.decoder(x)
        B, _, H, W = x.size()
        return x


class Residual(nn.Module):
    def __init__(self, channels, us=False, ds=False):
        super(Residual, self).__init__()
        self.us = us
        self.ds = ds

        layers = [
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True)
        ]
        if ds:
            layers += [
                nn.Conv2d(channels, channels, 1, bias=False, stride=2),
                nn.BatchNorm2d(channels)
            ]
        elif us:
            layers += [
                nn.ConvTranspose2d(channels, channels, 1, bias=False, stride=2),
                nn.BatchNorm2d(channels)
            ]
        else:
            layers += [
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels)
            ]

        self.block = nn.Sequential(*layers)


    def forward(self, x):
        input = x
        out = self.block(x)
        if self.us or self.ds:
            input = F.upsample(x, size=out.shape[2:])

        return input + out


# Inference Network
class InferenceNet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(InferenceNet, self).__init__()

    # q(y|x)
    '''
    self.inference_qyx = torch.nn.ModuleList([
        nn.Linear(x_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        GumbelSoftmax(512, y_dim)
    ])
    '''

    self.trunk = nn.Sequential(
            Encoder(128, 16),
            nn.Flatten(1)
    )


    self.inference_qyx = torch.nn.ModuleList([
        nn.Linear(16 * 4 * 4, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        GumbelSoftmax(512, y_dim)
    ])

    # q(z|y,x)
    self.inference_qzyx = torch.nn.ModuleList([
        nn.Linear(16 * 4 * 4 + y_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        Gaussian(512, z_dim)
    ])

  # q(y|x)
  def qyx(self, x, temperature, hard):
    num_layers = len(self.inference_qyx)
    for i, layer in enumerate(self.inference_qyx):
      if i == num_layers - 1:
        #last layer is gumbel softmax
        x = layer(x, temperature, hard)
      else:
        x = layer(x)
    return x

  # q(z|x,y)
  def qzxy(self, x, y):
    concat = torch.cat((x, y), dim=1)
    for layer in self.inference_qzyx:
      concat = layer(concat)
    return concat

  def forward(self, x, temperature=1.0, hard=0):
    if x.ndim == 2:
        x = x.reshape(-1, 1, 28, 28)

    #x = Flatten(x)
    raw_x = x
    x = self.trunk(x)

    # q(y|x)
    logits, prob, y = self.qyx(x, temperature, hard)

    # q(z|x,y)
    mu, var, z = self.qzxy(x, y)

    output = {'mean': mu, 'var': var, 'gaussian': z,
              'logits': logits, 'prob_cat': prob, 'categorical': y}
    return output

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

# Generative Network
class GenerativeNet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(GenerativeNet, self).__init__()

    # p(z|y)
    self.y_mu = nn.Linear(y_dim, z_dim)
    self.y_var = nn.Linear(y_dim, z_dim)

    # p(x|z)
    '''
    self.generative_pxz = torch.nn.ModuleList([
        nn.Linear(z_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, x_dim),
        torch.nn.Sigmoid()
    ])
    '''
    self.generative_pxz = nn.Sequential(
        nn.Linear(z_dim, 4 * 4 * 16),
        View((-1, 16, 4, 4)),
        Decoder(128, 16),
        nn.Sigmoid(),
        nn.Flatten(1)
    )


  # p(z|y)
  def pzy(self, y):
    y_mu = self.y_mu(y)
    y_var = F.softplus(self.y_var(y))
    return y_mu, y_var

  # p(x|z)
  def pxz(self, z):
    return self.generative_pxz(z)

  def forward(self, z, y):
    # p(z|y)
    y_mu, y_var = self.pzy(y)

    # p(x|z)
    x_rec = self.pxz(z)

    output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec}
    return output


# GMVAE Network
class GMVAENet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(GMVAENet, self).__init__()

    self.inference = InferenceNet(x_dim, z_dim, y_dim)
    self.generative = GenerativeNet(x_dim, z_dim, y_dim)

    # weight initialization
    for m in self.modules():
      if m._modules is None:
          continue

      if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
          init.constant_(m.bias, 0)

  def forward(self, x, temperature=1.0, hard=0):
    x = x.view(x.size(0), -1)
    out_inf = self.inference(x, temperature, hard)
    z, y = out_inf['gaussian'], out_inf['categorical']
    out_gen = self.generative(z, y)

    # merge output
    output = out_inf
    for key, value in out_gen.items():
      output[key] = value
    return output
