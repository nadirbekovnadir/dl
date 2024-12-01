import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.profilers as pl_profilers

from model import NNModule
from dataset import Cifar10DataModule
import config
import torch
import tensorboardX


if __name__ == "__main__":
    logs_dir = "cifar-10/.logs"
    model_name = 'base'

    logger = pl_loggers.TensorBoardLogger(logs_dir, name=model_name)
    profiler = pl_profilers.PyTorchProfiler(
        activites=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(logs_dir + '/profiler0'),
        record_shapes=True,
        trace_memory=True,
        scheduler=torch.profiler.schedule(skip_first=1, wait=1, warmup=1, active=2)
    )

    nn_module = NNModule(
        num_classes=10,
        lr=config.LR,
    )

    datamodule = Cifar10DataModule(
        config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    trainer = pl.Trainer(
        profiler=profiler,
        accelerator=config.ACCELERATOR,
        strategy=config.STRATEGY,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.EPOCHS,
        precision=config.PRECISION,
        val_check_interval=1.0,
        logger=logger,
    )

    trainer.fit(nn_module, datamodule)
    trainer.test(nn_module, datamodule)
