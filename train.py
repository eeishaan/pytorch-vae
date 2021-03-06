#!/usr/bin/env python3
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
from torch.optim import Adam

from loader import get_loaders
from vae_model import VAE


def compute_loss(inputs, outputs, mu, logvar):
    reconstruction_loss = nn.MSELoss(reduction='sum')(inputs, outputs)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return kl_loss + reconstruction_loss


def train_vae():

    batch_size = 64
    epochs = 100
    latent_dimension = 100
    patience = 10

    device = torch.device('cuda:0') \
        if torch.cuda.is_available() \
        else torch.device('cpu')

    # load data
    train_loader, valid_loader = get_loaders('data', batch_size)

    model = VAE(latent_dimension).to(device)

    optim = Adam(model.parameters(), lr=1e-3)

    # intialize variables for early stopping
    val_greater_count = 0
    last_val_loss = 0

    for e in range(epochs):
        running_loss = 0
        model.train()
        for _, (images, _) in enumerate(train_loader):
            images = images.to(device)
            model.zero_grad()
            outputs, mu, logvar = model(images)
            loss = compute_loss(images, outputs, mu, logvar)
            running_loss += loss
            loss.backward()
            optim.step()

        running_loss = running_loss/len(train_loader)
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for images, _ in valid_loader:
                images = images.to(device)
                outputs, mu, logvar = model(images)
                loss = compute_loss(images, outputs, mu, logvar)
                val_loss += loss
            val_loss /= len(valid_loader)

        # increment variable for early stopping
        if val_loss > last_val_loss:
            val_greater_count += 1
        else:
            val_greater_count = 0
        last_val_loss = val_loss

        # save model
        torch.save({
            'epoch': e,
            'model': model.state_dict(),
            'running_loss': running_loss,
            'optim': optim.state_dict(),
        }, "checkpoint_{}.pth".format(e))
        print("Epoch: {} Train Loss: {}".format(e+1, running_loss.item()))
        print("Epoch: {} Val Loss: {}".format(e+1, val_loss.item()))

        # check early stopping condition
        if val_greater_count >= patience:
            break


if __name__ == '__main__':
    train_vae()
