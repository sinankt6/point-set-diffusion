import os
from pathlib import Path

from ps_diff.data import DataModule
from ps_diff.diffusion.model import PSDiff
from ps_diff.backbones.classifier import PointClassifier
from ps_diff.distributions.intensities import MixtureIntensity
from ps_diff.lightning_tasks import DensityEstimation


def instantiate_datamodule(config, seed):
    # Convert config to OmegaConf DictConfig
    if config["data_type"] == "tpp":
        root = Path(os.path.abspath(os.path.join(config["root"], "TPP")))
    else:
        root = Path(os.path.abspath(os.path.join(config["root"], config["name"])))

    return DataModule(
        root=root,
        name=config["name"],
        data_type=config["data_type"],
        batch_size=config["batch_size"],
        split_seed=seed,
    )


def instantiate_model(config, datamodule) -> PSDiff:
    dim = datamodule.dataset.space_bound.shape[0]

    classifier = PointClassifier(
        hidden_dims=config["hidden_dims"],
        layer=config["classifier_layer"],
    )
    intensity = MixtureIntensity(
        dim=dim,
        n_components=config["mix_components"],
        embedding_size=config["hidden_dims"],
        distribution="multivar_normal",
    )

    model = PSDiff(
        classifier_model=classifier,
        intensity_model=intensity,
        space_bound=datamodule.dataset.space_bound,
        n_max=datamodule.n_max,
        steps=config["steps"],
        hpp_scale=datamodule.n_mean / (2**dim),
        emb_dim=config["hidden_dims"],
        encoder_n_blocks=config["encoder_n_blocks"],
    )
    return model


def instantiate_task(config, model):
    if config["name"] == "density":
        return DensityEstimation(
            model=model,
            learning_rate=config["optimizer"]["learning_rate"],
            lr_decay=config["optimizer"]["lr_decay"],
            weight_decay=config["optimizer"]["weight_decay"],
            lr_schedule=config["optimizer"]["lr_schedule"],
            point_process_type=config["point_process_type"],
        )
