from typing import Any
import torch
import torchvision
import torchvision.transforms.v2
import pytorch_lightning as pl


class Cifar10DataModule(pl.LightningDataModule):
    def __init__(self, data_save_dir, batch_size, num_workers):
        super().__init__()
        self.data_save_dir = data_save_dir
        self.data_train_dir = data_save_dir + "/train"
        self.data_test_dir = data_save_dir + "/test"

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        torchvision.datasets.CIFAR10(
            root=self.data_train_dir,
            download=True,
            train=True,
        )
        torchvision.datasets.CIFAR10(
            root=self.data_test_dir,
            download=True,
            train=False,
        )

    def setup(self, stage: str) -> None:
        transform = torchvision.transforms.v2.Compose(
            [
                torchvision.transforms.v2.ToImage(),
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                torchvision.transforms.v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.train_ds = torchvision.datasets.CIFAR10(
            root=self.data_train_dir, download=False, train=False, transform=transform
        )

        self.train_ds, self.val_ds = torch.utils.data.random_split(
            self.train_ds, lengths=[0.8, 0.2]
        )

        self.test_ds = torchvision.datasets.CIFAR10(
            root=self.data_test_dir, download=False, train=False, transform=transform
        )

    def train_dataloader(self) -> Any:
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> Any:
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> Any:
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
