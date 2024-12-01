import sys
import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.profilers as pl_profilers

import torch
import tensorboardX


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: OmegaConf):
    print(OmegaConf.to_yaml(cfg))

    torch.set_float32_matmul_precision("high")

    logs_dir = "cifar_10/.logs"
    model_name = "base"

    logger = pl_loggers.TensorBoardLogger(logs_dir, name=model_name)
    profiler = pl_profilers.PyTorchProfiler(
        activites=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            logs_dir + "/profiler0"
        ),
        record_shapes=True,
        trace_memory=True,
        scheduler=torch.profiler.schedule(skip_first=1, wait=1, warmup=1, active=2),
    )

    datamodule = hydra.utils.instantiate(cfg.data_module)
    nn_module = hydra.utils.instantiate(cfg.nn_module)

    trainer = pl.Trainer(
        profiler=profiler,
        accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.strategy,
        devices=cfg.trainer.devices,
        min_epochs=1,
        max_epochs=cfg.hp.epochs,
        precision=cfg.trainer.precision,
        val_check_interval=1.0,
        logger=logger,
    )

    trainer.fit(nn_module, datamodule)
    trainer.test(nn_module, datamodule)


if __name__ == "__main__":
    main()
