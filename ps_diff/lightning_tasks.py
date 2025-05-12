import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from ps_diff.data import Batch

from ps_diff.density_tasks import get_metrics_stpp, get_metrics_spp, log_metrics
from ps_diff.metrics_add_thin import (
    lengths_distribution_wasserstein_distance,
    MMD,
)


class Tasks(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate,
        lr_decay: float,
        weight_decay: float = 0.0,
        lr_schedule=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("model",))

        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate

        self.model = model
        self.classification_loss_func = nn.BCEWithLogitsLoss(reduction="none")

    def classification_loss(self, x_n_int_x_0, x_n: Batch):
        """
        Compute BCE loss for the classification task.
        """
        x_n_int_x_0 = x_n_int_x_0.flatten()[x_n.mask[:, :, 0].flatten()]
        target = x_n.kept.flatten()[x_n.mask[:, :, 0].flatten()]
        loss = self.classification_loss_func(x_n_int_x_0, target.float())
        loss = (loss).sum() / len(x_n)
        return loss

    def intensity_loss(self, log_prob_x_0):
        """
        Compute the average (over batch) negative log-likelihood of the event sequences.
        """
        return -log_prob_x_0.mean()

    def get_loss(self, log_prob_x_0, x_n_int_x_0, x_n):
        """
        Compute the loss for the classification and intensity.
        """
        intensity = self.intensity_loss(log_prob_x_0) / self.model.n_max

        classification = self.classification_loss(x_n_int_x_0, x_n) / self.model.n_max
        loss = classification + intensity

        return loss, classification, intensity

    def step(self, batch, name):
        """
        Apply model to batch and compute loss.
        """
        # Forward pass
        x_n_int_x_0, log_prob_x_0, x_n = self.model.forward(batch)

        # Compute loss
        loss, classification, intensity = self.get_loss(log_prob_x_0, x_n_int_x_0, x_n)

        # Log loss
        self.log(
            f"{name}/loss",
            loss.detach().item(),
            batch_size=batch.batch_size,
            on_epoch=True,
        )
        self.log(
            f"{name}/log-likelihood",
            intensity.detach().item(),
            batch_size=batch.batch_size,
            on_epoch=True,
        )
        if classification is not None:
            self.log(
                f"{name}/BCE",
                classification.detach().item(),
                batch_size=batch.batch_size,
                on_epoch=True,
            )

        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=50, verbose=True
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "train/loss",
            },
        }


class DensityEstimation(Tasks):
    def __init__(
        self,
        model,
        learning_rate,
        lr_decay,
        weight_decay,
        lr_schedule,
        point_process_type,
    ):
        super().__init__(model, learning_rate, lr_decay, weight_decay, lr_schedule)

        assert point_process_type in [
            "spp",
            "stpp",
            "tpp",
        ], "Unknown point process type."
        self.point_process_type = point_process_type

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, "train")
        return {"loss": loss}

    def evaluation_step(self, batch, split: str):
        if split == "test" and self.point_process_type == "tpp":
            n_samples = 4000
        elif split == "val":
            n_samples = 100
        else:
            n_samples = 1000

        target = batch.to_point_list()
        sample = self.model.sample(n_samples, space_bound=batch.space_bound)
        # Save for later
        self.sampled_batch = sample

        # Convert to point list
        sample = sample.to_point_list()

        # Set space bound to be the same as the normalized batch
        space_bound = np.ones_like(batch.space_bound.detach().cpu().numpy())
        space_bound[:, 0] = space_bound[:, 0] * -1

        # Compute metrics
        if self.point_process_type == "spp":
            metric_dict = get_metrics_spp(
                sample,
                target,
                space_bound,
                self.model.n_max,
                split == "test",
            )
            # Log metrics
            log_metrics(metric_dict, self, split, batch.batch_size)
        elif self.point_process_type == "stpp":
            metric_dict = get_metrics_stpp(
                sample,
                target,
                space_bound,
                self.model.n_max,
            )
            # Log metrics
            log_metrics(metric_dict, self, split, batch.batch_size)
        elif self.point_process_type == "tpp":
            tmax = (
                (batch.space_bound[0, 1] - batch.space_bound[0, 0])
                .detach()
                .cpu()
                .numpy()
            )
            for i, _ in enumerate(sample):
                sample[i] = tmax * (sample[i].squeeze(1) + 1) / 2

            for i, _ in enumerate(target):
                target[i] = tmax * (target[i].squeeze(1) + 1) / 2

            mmd = MMD(
                sample,
                target,
                tmax,
            )[0]
            wasserstein = lengths_distribution_wasserstein_distance(
                sample,
                target,
                tmax,
                self.model.n_max,
            )
            self.log(f"{split}/sample_mmd", mmd, batch_size=batch.batch_size)
            self.log(
                f"{split}/sample_count_wasserstein",
                wasserstein,
                batch_size=batch.batch_size,
            )

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            self.evaluation_step(batch, "test")

    def validation_step(self, batch, batch_idx):
        if self.global_step > 1 and self.current_epoch > 50:
            with torch.no_grad():
                self.evaluation_step(batch, "val")

        else:
            self.log(
                f"val/mmd_counting_stpp",
                1,
                batch_size=batch.batch_size,
                on_epoch=True,
            )
            self.log(
                f"val/count_wasserstein",
                1,
                batch_size=batch.batch_size,
                on_epoch=True,
            )
            self.log(
                f"val/sample_mmd",
                1,
                batch_size=batch.batch_size,
                on_epoch=True,
            )
