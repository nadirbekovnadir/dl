from typing import Any, Mapping, Tuple
import pytorch_lightning as pl
import torch
import torchvision
import torchmetrics
import torchvision.transforms.v2


class NNModule(pl.LightningModule):
    def __init__(self, num_classes=2) -> None:
        super().__init__()

        self.model = torchvision.models.efficientnet_b0(num_classes=num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task='multiclass', num_classes=num_classes)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=0.001)
    
    def _base_step(self, batch, batch_idx) -> Tuple[torch.Tensor]:
        x, target = batch
        preds = self.forward(x)

        loss = self.loss_fn(preds, target)
        return loss, preds, target
    
    def on_train_epoch_end(self) -> None:
        return super().on_train_epoch_end()
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss, preds, target = self._base_step(batch, batch_idx)
        accuracy = self.accuracy(preds, target)
        f1_score = self.f1_score(preds, target)
        self.log_dict(
            {'train_loss': loss, 'train_accuracy': accuracy, 'train_f1_score': f1_score},
            prog_bar=True, on_epoch=True, on_step=False,
        )
        return loss
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss, preds, target = self._base_step(batch, batch_idx)
        accuracy = self.accuracy(preds, target)
        f1_score = self.f1_score(preds, target)
        self.log_dict(
            {'val_loss': loss, 'val_accuracy': accuracy, 'val_f1_score': f1_score},
            prog_bar=True, on_epoch=True, on_step=False,
        )
        return loss
    
    def test_step(self, batch, batch_idx) -> torch.Tensor:
        loss, preds, target = self._base_step(batch, batch_idx)
        accuracy = self.accuracy(preds, target)
        f1_score = self.f1_score(preds, target)
        self.log_dict(
            {'test_loss': loss, 'test_accuracy': accuracy, 'test_f1_score': f1_score},
            prog_bar=True, on_epoch=True, on_step=False,
        )
        return loss
    

class Cifar10DataModule(pl.LightningDataModule):
    def __init__(self, data_save_dir, batch_size, num_workers):
        super().__init__()
        self.data_save_dir = data_save_dir
        self.data_train_dir = data_save_dir + '/train'
        self.data_test_dir = data_save_dir + '/test'

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
        transform = torchvision.transforms.v2.Compose([
            torchvision.transforms.v2.ToImage(), 
            torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
            torchvision.transforms.v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.train_ds = torchvision.datasets.CIFAR10(
            root=self.data_train_dir, 
            download=False,
            train=False,
            transform=transform
        )

        self.train_ds, self.val_ds = torch.utils.data.random_split(
            self.train_ds, lengths=[0.8, 0.2]
        )

        self.test_ds = torchvision.datasets.CIFAR10(
            root=self.data_test_dir, 
            download=False, 
            train=False, 
            transform=transform
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


