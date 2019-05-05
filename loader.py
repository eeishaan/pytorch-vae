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

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
from torchvision.datasets import SVHN


def get_loaders(data_dir, batch_size, split=0.9):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5),
                             (.5, .5, .5))
    ])

    dataset = SVHN(data_dir,
                   split='train',
                   download=True,
                   transform=transform)

    train_len = int(len(dataset) * split)
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    valid_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4)

    return train_loader, valid_loader
