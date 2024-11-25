import pytorch_lightning as pl

from modules import Cifar10DataModule, NNModule


if __name__ == '__main__':
    nn_module = NNModule(num_classes=10)
    datamodule = Cifar10DataModule(
        './cifar-10/.data', 
        batch_size=64, 
        num_workers=8
    )

    trainer = pl.Trainer(
        accelerator='gpu', 
        strategy='auto',
        devices=[0],
        min_epochs=1,
        max_epochs=3,
        precision=16,
        val_check_interval=1.0,
    )

    trainer.fit(nn_module, datamodule)
    trainer.test(nn_module, datamodule)