import hydra
from omegaconf import OmegaConf
import torch
import torchvision
import torchmetrics
import pytorch_lightning as pl
from typing import Tuple


class NNModule(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        model,
        loss,
        optimizer_builder,
    ) -> None:
        super().__init__()

        self.training_outputs = []
        self.validation_outputs = []

        self.model = model
        self.loss_fn = loss
        self.optimizer_builder = optimizer_builder

        self.accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        return self.optimizer_builder(self.model.parameters())

    def _base_step(self, batch, batch_idx) -> Tuple[torch.Tensor]:
        x, target = batch
        preds = self.forward(x)

        loss = self.loss_fn(preds, target)
        return loss, preds, target

    def calculate_metrics(self, stage, data):
        loss = torch.cat([r["loss"][None] for r in data])
        preds = torch.cat([r["preds"] for r in data])
        target = torch.cat([r["target"] for r in data])

        accuracy = self.accuracy(preds, target)
        f1_score = self.f1_score(preds, target)
        self.log_dict(
            {
                f"{stage}_loss_mean": loss.mean(),
                f"{stage}_loss_std": loss.std(),
                f"{stage}_accuracy": accuracy,
                f"{stage}_f1_score": f1_score,
            },
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss, preds, target = self._base_step(batch, batch_idx)
        if self.global_step % 100 == 0:
            x = batch[0][:8]
            grid = torchvision.utils.make_grid(x)
            self.logger.experiment.add_image(
                "training_images",
                grid,
                self.global_step,
            )

        self.training_outputs.append({"loss": loss, "preds": preds, "target": target})
        return loss

    def on_train_epoch_end(self):
        self.calculate_metrics("train", self.training_outputs)

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss, preds, target = self._base_step(batch, batch_idx)
        self.validation_outputs.append({"loss": loss, "preds": preds, "target": target})
        return loss

    def on_validation_epoch_end(self):
        self.calculate_metrics("val", self.validation_outputs)

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        loss, preds, target = self._base_step(batch, batch_idx)
        accuracy = self.accuracy(preds, target)
        f1_score = self.f1_score(preds, target)
        self.log_dict(
            {"test_loss": loss, "test_accuracy": accuracy, "test_f1_score": f1_score},
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        return loss
