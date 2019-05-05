# Copyright (C) 2019  Ishaan Kumar

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, base_channel):
        super().__init__()
        self.base_channel = base_channel
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=self.base_channel,
                kernel_size=4, padding=1, stride=2),  # 16
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels=self.base_channel,
                out_channels=self.base_channel*2,
                kernel_size=4, padding=1, stride=2),  # 8
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels=self.base_channel*2,
                out_channels=self.base_channel*4,
                kernel_size=4, padding=1, stride=2),  # 4
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels=self.base_channel*4,
                out_channels=self.base_channel*8,
                kernel_size=4, padding=1, stride=2),  # 2
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpsampleDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        base_channel = 64
        self.network = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(
                in_channels=latent_dim,
                out_channels=base_channel*8,
                bias=False,
                kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_channel*8),
            nn.ReLU(True),  # 4

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=base_channel*8,
                      out_channels=base_channel*4,
                      bias=False,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_channel*4),
            nn.ReLU(True),  # 8

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=base_channel*4,
                      out_channels=base_channel*2,
                      bias=False,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_channel*2),
            nn.ReLU(True),  # 16

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=base_channel*2,
                      out_channels=3,
                      kernel_size=3, padding=1),
            nn.Sigmoid()  # 32
        )

    def forward(self, x):
        return self.network(x.unsqueeze(-1).unsqueeze(-1))


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        base_channel = 64
        self.lin_in_dim = 2*2*base_channel*8

        # define encoder block
        self.encoder = EncoderBlock(base_channel)

        self.lin1 = nn.Sequential(
            nn.Linear(self.lin_in_dim, latent_dim),
            nn.ReLU(),
        )

        # linear layers for mu and logvar prediction
        self.lin11 = nn.Linear(latent_dim, latent_dim)
        self.lin12 = nn.Linear(latent_dim, latent_dim)

        # decoder block
        self.decoder = UpsampleDecoder(latent_dim)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, x):
        z = self.encoder(x)
        z = z.view(-1, self.lin_in_dim)
        z = self.lin1(z)
        mu = self.lin11(z)
        logvar = self.lin12(z)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar
