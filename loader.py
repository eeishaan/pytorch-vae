from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
from torchvision.datasets import SVHN


def get_loaders(data_dir, batch_size, split=0.9):
    transform = transforms.ToTensor()

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

    test_loader = DataLoader(
        SVHN(data_dir,
             split='test',
             download=True,
             transform=transform),
        batch_size=batch_size)

    return train_loader, valid_loader, test_loader
