import os
import torchvision

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from typing import Tuple
from torchvision.transforms import Compose, ToTensor, Normalize


def get_dataloaders(
    cfg: DictConfig, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Return training and validation dataloaders"""

    ##################################################################
    # Change this. Demo dataset is MNIST
    ##################################################################
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    dataset = torchvision.datasets.MNIST(
        os.path.join(os.environ["ORIG_CWD"], cfg.dirs.data),
        download=True,
        transform=data_transform,
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=cfg.mode.train.batch_size,
        shuffle=cfg.mode.train.shuffle,
        num_workers=num_workers,
    )

    val_dataloader = DataLoader(
        dataset,
        batch_size=cfg.mode.val.batch_size,
        shuffle=cfg.mode.val.shuffle,
        num_workers=num_workers,
    )

    return train_dataloader, val_dataloader
