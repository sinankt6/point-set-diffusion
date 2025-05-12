import os
import pickle
from typing import List, Sequence, Tuple

import numpy as np
from omegaconf import OmegaConf
from ps_diff.config import instantiate_datamodule, instantiate_model
from ps_diff.lightning_tasks import DensityEstimation


def load_task(wandb_run_dir) -> DensityEstimation:
    # Load config file from wandb run
    config_dir = os.path.abspath(
        os.path.join(
            "logs", "wandb", wandb_run_dir, "files", "config_seml.yaml"
        )
    )
    config = OmegaConf.load(config_dir)
    config["task_params"]["name"] = "forecast"

    datamodule_params = config.get("datamodule_params")
    seed = config.get("seed")
    datamodule = instantiate_datamodule(datamodule_params, seed)
    datamodule.prepare_data()

    model_params = config.get("model_params")
    model = instantiate_model(model_params, datamodule)

    task_dir = os.path.abspath(
        os.path.join(
            "logs", "wandb", wandb_run_dir, "files", "checkpoints", "best.ckpt"
        )
    )
    best_task = DensityEstimation.load_from_checkpoint(task_dir, model=model)
    return best_task, datamodule, seed, datamodule_params, model_params


def find_run_dir(run_id):
    wandb_run_dir = None
    for root, dirs, files in os.walk("logs/wandb"):
        for dir in dirs:
            if run_id in dir:
                wandb_run_dir = dir
                break
        if wandb_run_dir:
            break
    return wandb_run_dir


def load_data_model(
    process: str, model, task="unconditional"
) -> Tuple[List[Sequence], List[Sequence]]:
    files = os.listdir(f"results/{task}/{model}/")
    data = dict()
    for file in files:
        with open(f"results/{task}/{model}/{file}", "rb") as f:
            print(f"results/{task}/{model}/{file}")

            split_file_name = file.split("_")

            for i, split_str in enumerate(split_file_name):
                if "seed" in split_str:
                    seed = split_str.replace("seed", "")
                    break

            print(seed, i, split_file_name)

            dataset = ("_").join(file.split("_")[i + 1 :]).replace(".pkl", "")

            if dataset not in data:
                data[dataset] = {seed: pickle.load(f)}
            else:
                data[dataset][seed] = pickle.load(f)

    return data
