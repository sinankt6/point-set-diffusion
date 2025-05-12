import logging
import faulthandler
import pickle
import warnings
import sys
import pytorch_lightning as pl
import wandb
import os
import hydra

from hydra.utils import instantiate
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from ps_diff.config import (
    instantiate_datamodule,
    instantiate_model,
    instantiate_task,
)

from ps_diff.utils import (
    get_logger,
    print_config,
    log_hyperparameters,
    filter_device_available,
    print_exceptions,
    WandbSummaries,
    WandbModelCheckpoint,
)

sys.path.append(".")


def get_callbacks(task_params):
    monitor = {"monitor": task_params["metric"], "mode": "min"}
    callbacks = [
        WandbSummaries(**monitor),
        WandbModelCheckpoint(
            save_last=True,
            save_top_k=1,
            every_n_epochs=1,
            filename="best",
            **monitor,
        ),
        TQDMProgressBar(refresh_rate=1),
    ]

    if task_params["early_stopping"] is not None:
        stopper = EarlyStopping(
            patience=int(task_params["early_stopping"]),
            min_delta=0,
            strict=False,
            check_on_train_epoch_end=False,
            **monitor,
        )
        callbacks.append(stopper)
    return callbacks


# Log to traceback to stderr on segfault
faulthandler.enable(all_threads=False)

# Stop lightning from pestering us about things we already know
warnings.filterwarnings(
    "ignore",
    "There is a wandb run already in progress",
    module="pytorch_lightning.loggers.wandb",
)
warnings.filterwarnings(
    "ignore",
    "The dataloader, [^,]+, does not have many workers",
    module="pytorch_lightning",
)
logging.getLogger("pytorch_lightning.utilities.rank_zero").addFilter(
    filter_device_available
)

log = get_logger()

print("Starting script")


@hydra.main(config_path="config", config_name="spp_train", version_base=None)
@print_exceptions
def main(config: DictConfig):
    OmegaConf.resolve(config)

    # Start wandb
    wandb.init(
        project=f"ps_diff_{config.task_params.point_process_type}_density",
        name=f"ps_diff_{config.task_params.point_process_type}_{config.datamodule.name}_{config.seed}",
        dir=Path(os.path.abspath("logs")),
    )

    OmegaConf.save(config, wandb.run.dir + "/config_hydra.yaml")
    log.info(wandb.run.dir)

    log.info(f"Setting seed {config.seed}")
    pl.seed_everything(config.seed)

    # Save config on wandb
    OmegaConf.save(config, wandb.run.dir + "/config_seml.yaml")
    log.info(wandb.run.dir)

    log.info("Loading data")
    print_config(config.datamodule)
    datamodule = instantiate_datamodule(config.datamodule, config.seed)
    datamodule.prepare_data()

    log.info(config.datamodule["name"])
    log.info(f"Average number of point sequences: {datamodule.n_mean}")

    log.info("Instantiating model")
    model = instantiate_model(config.model_params, datamodule)

    log.info("Instantiating task")

    task = instantiate_task(config.task_params, model)

    logger = WandbLogger()
    # log hyperparameters to wandb logger
    log_hyperparameters(logger, config, model)

    log.info("Loading checkpoint")
    # Get all required callbacks
    callbacks = get_callbacks(config.task_params)

    log.info("Instantiating trainer")

    # Get trainer
    trainer: Trainer = instantiate(
        config.trainer,
        accelerator="gpu" if config.use_gpu else "cpu",
        callbacks=callbacks,
        logger=logger,
    )

    log.info("Trainer loaded")

    trainer.fit(task, datamodule=datamodule)

    if config.test:
        trainer.test(task, datamodule=datamodule, ckpt_path="best")

        # Save test samples
        log.info(f"Saving test samples to {Path(wandb.run.dir) / 'test_samples.pkl'}")
        with open(Path(wandb.run.dir) / "test_samples.pkl", "wb") as handle:
            pickle.dump(task.sampled_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)

    wandb.finish()

    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    best_score = trainer.checkpoint_callback.best_model_score
    return float(best_score) if best_score is not None else None


if __name__ == "__main__":
    main()
