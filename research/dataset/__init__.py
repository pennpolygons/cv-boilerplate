import torchvision

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from typing import Tuple


def get_dataloaders(
    cfg: DictConfig, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Return training and validation dataloaders"""

    ##################################################################
    # Change this. Demo dataset is MNIST
    ##################################################################
    dataset = torchvision.datasets.MNIST(cfg.dirs.data, download=True)

    train_dataloader = DataLoader(
        dataset,
        batch_size=cfg.mode.train.batch_size,
        shuffle=cfg.mode.train.shuffle,
        num_workers=num_workers,
        worker_init_fn=lambda _: cfg.random_seed,
    )

    val_dataloader = DataLoader(
        dataset,
        batch_size=cfg.mode.val.batch_size,
        shuffle=cfg.mode.val.shuffle,
        num_workers=num_workers,
        worker_init_fn=lambda _: cfg.random_seed,
    )

    return train_dataloader, val_dataloader
