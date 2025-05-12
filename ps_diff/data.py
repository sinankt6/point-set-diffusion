import os
import re
from pathlib import Path
from typing import List, Tuple, Union
from statistics import mean

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

SPACE_BOUNDARIES = {
    "earthquakes_spp": np.array([[122.0, 150.2], [21.7, 46.3]]),
    "earthquakes_stpp": np.array([[0.0, 30.0], [122.0, 150.2], [21.7, 46.3]]),
    "covid_nj_cases_spp": np.array([[-75.6, -73.7], [38.7, 41.4]]),
    "covid_nj_cases_stpp": np.array([[0.0, 7.0], [-75.6, -73.7], [38.7, 41.4]]),
    "citibike_spp": np.array([[-74.04, -73.84], [40.65, 40.87]]),
    "citibike_stpp": np.array([[0.0, 24.0], [-74.04, -73.84], [40.65, 40.87]]),
    "pinwheel_spp": np.array([[-4.20, 4.79], [-4.44, 4.04]]),
    "pinwheel_stpp": np.array([[0.0, 30.0], [-4.20, 4.79], [-4.44, 4.04]]),
}

LIMITS_MAPS = {
    "earthquakes": np.array([[123.43, 149.18], [25.41, 45.98]]),
    "covid_nj_cases": np.array([[-75.60, -73.90], [38.90, 41.20]]),
    "citibike": np.array([[-74.03, -73.87], [40.65, 40.87]]),
}

MAPS = {
    "citibike": "data/maps/manhattan_map.png",
    "covid_nj_cases": "data/maps/nj_map.png",
    "earthquakes": "data/maps/jp_map.png",
    "crime": None,
    "bold5000": None,
    "pinwheel": None,
    "independent": None,
}


def denormalize_points(
    points: List[np.ndarray], space_bound: np.ndarray, norm_space_bound: np.ndarray
) -> np.ndarray:
    denormalized_points = []

    for point in points:
        denormalized_points.append(
            (
                (point - norm_space_bound[:, 0][None, :])
                / (norm_space_bound[:, 1][None, :] - norm_space_bound[:, 0][None, :])
                * (space_bound[:, 1][None, :] - space_bound[:, 0][None, :])
                + space_bound[:, 0][None, :]
            )
        )

    return denormalized_points


@typechecked
class Sequence:
    def __init__(
        self,
        points: Union[np.ndarray, TensorType[float, "events", "dim"]],
        space_bound: Union[np.ndarray, TensorType[float, "dim", 2]],
        device: Union[torch.device, str] = "cpu",
        kept_points: Union[np.ndarray, TensorType, None] = None,
    ) -> None:
        super().__init__()
        if not isinstance(points, torch.Tensor):
            points = torch.as_tensor(points, dtype=torch.float32)

        if space_bound is not None:
            if not isinstance(space_bound, torch.Tensor):
                space_bound = torch.as_tensor(space_bound, dtype=torch.float32)

        if kept_points is not None:
            if not isinstance(kept_points, torch.Tensor):
                kept_points = torch.as_tensor(kept_points, dtype=torch.float32)
            kept_points = kept_points

        self.points = points
        self.space_bound = space_bound
        self.kept_points = kept_points

        self.device = device
        self.to(device)

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, key: str):
        return getattr(self, key, None)

    def __setitem__(self, key: str, value):
        setattr(self, key, value)

    def keys(self) -> List[str]:
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != "__" and key[-2:] != "__"]
        return keys

    def __iter__(self):
        for key in sorted(self.keys()):
            yield key, self[key]

    def __contains__(self, key):
        return key in self.keys()

    def to(self, device: Union[str, torch.device]) -> "Sequence":
        self.device = device
        for key in self.keys():
            if key != "device":
                self[key] = self[key].to(device)
        return self


@typechecked
class Batch:
    def __init__(
        self,
        mask: TensorType[bool, "batch", "sequence", "dim"],
        points: TensorType[float, "batch", "sequence", "dim"],
        space_bound: TensorType[float, "dim", 2],
        unpadded_length: TensorType[int, "batch"],
        kept: Union[TensorType[bool, "batch", "sequence"], None] = None,
    ):
        super().__init__()
        self.points = points
        self.space_bound = space_bound
        self.kept = kept
        self.dim = space_bound.shape[0]

        # Padding and mask
        self.unpadded_length = unpadded_length
        self.mask = mask

        self._validate()

    @property
    def batch_size(self) -> int:
        return self.points.shape[0]

    @property
    def seq_len(self) -> int:
        return self.points.shape[1]

    def __len__(self):
        return self.batch_size

    def __getitem__(self, key: str):
        return getattr(self, key, None)

    def __setitem__(self, key: str, value):
        setattr(self, key, value)

    def keys(self) -> List[str]:
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != "__" and key[-2:] != "__"]
        return keys

    def __iter__(self):
        for key in sorted(self.keys()):
            yield key, self[key]

    def __contains__(self, key):
        return key in self.keys()

    def to(self, device: Union[str, torch.device]) -> "Batch":
        self.device = device
        for key in self.keys():
            if key != "device" and key != "dim":
                self[key] = self[key].to(device)
        return self

    @staticmethod
    def from_sequence_list(sequences: List[Sequence]) -> "Batch":
        """
        Create batch from list of sequences.
        """
        # Pad sequences for batching
        space_bound = torch.cat(
            [sequence.space_bound.unsqueeze(0) for sequence in sequences], dim=0
        )
        min_values, _ = torch.min(space_bound, dim=0)
        max_values, _ = torch.max(space_bound, dim=0)
        space_bound = torch.stack([min_values[:, 0], max_values[:, 1]], dim=1)
        points = pad([sequence.points for sequence in sequences])
        device = points.device
        dim, _ = space_bound.shape

        sequence_length = torch.tensor(
            [len(sequence) for sequence in sequences], device=device
        )

        if sequences[0].kept_points != None:
            kept_points = pad(
                [sequence.kept_points for sequence in sequences],
                length=points.shape[1],
            )
        else:
            kept_points = None

        # Compute event mask for batching
        mask = (
            torch.arange(0, points.shape[1], device=device)[None, :]
            < sequence_length[:, None]
        )
        mask = mask[..., None].repeat(1, 1, dim)

        batch = Batch(
            mask=mask,
            points=points,
            space_bound=space_bound,
            unpadded_length=sequence_length,
            kept=kept_points,
        )
        return batch

    def add_events(self, other: "Batch") -> "Batch":
        """
        Add batch of events to sequences.

        Parameters:
        ----------
        other : Batch
            Batch of events to add.

        Returns:
        -------
        Batch
            Batch of events with added events.
        """
        assert len(other) == len(
            self
        ), "The number of sequences to add does not match the number of sequences in the batch."
        other = other.to(self.points.device)
        min_values, _ = torch.min(
            torch.stack([self.space_bound[:, 0], other.space_bound[:, 0]], dim=1), dim=1
        )
        max_values, _ = torch.max(
            torch.stack([self.space_bound[:, 1], other.space_bound[:, 1]], dim=1), dim=1
        )
        space_bound = torch.stack([min_values, max_values], dim=1)

        if self.kept is None:
            kept = torch.cat(
                [
                    torch.ones_like(self.points[:, :, 0], dtype=bool)
                    * self.mask[:, :, 0],
                    torch.zeros_like(other.points[:, :, 0], dtype=bool),
                ],
                dim=1,
            )
        else:
            kept = torch.cat(
                [self.kept, torch.zeros_like(other.points[:, :, 0], dtype=bool)],
                dim=1,
            )

        return self.remove_unnescessary_padding(
            points=torch.cat([self.points, other.points], dim=1),
            mask=torch.cat([self.mask, other.mask], dim=1),
            kept=kept,
            space_bound=space_bound,
        )

    def to_point_list(self):
        point_list = []
        for i in range(len(self)):
            point_list.append(
                self.points[i][self.mask[i]]
                .detach()
                .cpu()
                .numpy()
                .reshape((self.mask[i, :, 0].sum(), self.dim))
            )
        return point_list

    def concat(self, *others):
        points = [self.points] + [o.points for o in others]
        mask = [self.mask] + [o.mask for o in others]
        return self.remove_unnescessary_padding(
            points=torch.cat(points, 0),
            mask=torch.cat(mask, 0),
            kept=None,
            space_bound=self.space_bound,
        )

    @staticmethod
    def sort_time(
        points: TensorType[float, "batch", "sequence", "dim"],
        mask: TensorType[bool, "batch", "sequence", "dim"],
        kept: Union[TensorType[bool, "batch", "sequence"], None],
        space_bound: TensorType[float, "dim", 2],
    ):
        """
        Sort events by first coordinate of point.

        Parameters:
        ----------
        points : TensorType[float, "batch", "sequence", "dim"]
            Tensor of event points.
        mask : TensorType[bool, "batch", "sequence", "dim"]
            Tensor of event masks.
        kept : Union[TensorType[bool, "batch", "sequence"], None]
            Tensor indicating kept events.
        space_bound : TensorType[float, "dim", 2]
            Lower and upper bounds of every dimension of the sample space.

        Returns:
        -------
        points : TensorType[float, "batch", "sequence", "dim"]
            Tensor of event points.
        mask : TensorType[bool, "batch", "sequence", "dim"]
            Tensor of event masks.
        kept : Union[TensorType[bool, "batch", "sequence"], None]
            Tensor indicating kept events.
        """
        # Sort points and mask by time (sup. first dimension)
        dim, _ = space_bound.shape
        # Use normalized space upper bound (1.0) to avoid getting masked values
        points[~mask] = 2 * torch.ones(dim).to(space_bound.device).repeat(
            int((points[~mask].shape)[0] / dim)
        )
        sort_idx = torch.argsort(points, dim=1)[:, :, 0].unsqueeze(2).repeat(1, 1, dim)
        mask = torch.take_along_dim(mask, sort_idx, dim=1)
        points = torch.take_along_dim(points, sort_idx, dim=1)
        if kept is not None:
            kept = torch.take_along_dim(kept, sort_idx[:, :, 0], dim=1)
        else:
            kept = None
        points = points * mask
        return points, mask, kept

    @staticmethod
    def remove_unnescessary_padding(
        points: TensorType[float, "batch", "sequence", "dim"],
        mask: TensorType[bool, "batch", "sequence", "dim"],
        kept: Union[TensorType[bool, "batch", "sequence"], None],
        space_bound: TensorType[float, "dim", 2],
    ):
        """
        Remove unnescessary padding from batch.

        Parameters:
        ----------
        points : TensorType[float, "batch", "sequence", "dim"]
            Tensor of event points.
        mask : TensorType[bool, "batch", "sequence", "dim"]
            Tensor of event masks.
        kept : Union[TensorType[bool, "batch", "sequence"], None]
            Tensor indicating kept events.
        space_bound : TensorType[float, "dim", 2]
            Lower and upper bounds of each dimension in the bounded space.

        Returns:
        -------
        Batch
            Batch of events without unnescessary padding.
        """
        # Sort by time (1st dim. of point process)
        points, mask, kept = Batch.sort_time(
            points, mask, kept, space_bound=space_bound
        )
        # Reduce padding along sequence length
        max_length = torch.max(
            mask.sum(dim=1)[:, 0]
        ).int()  # Based on first coord. of points mask
        mask = mask[:, : max_length + 1, :]
        points = points[:, : max_length + 1, :]
        if kept is not None:
            kept = kept[:, : max_length + 1]

        return Batch(
            mask=mask,
            points=points,
            space_bound=space_bound,
            unpadded_length=mask.sum(1)[:, 0].long(),
            kept=kept,
        )

    def thin(self, alpha: TensorType[float]) -> Tuple["Batch", "Batch"]:
        """
        Thin events according to alpha.

        Parameters:
        ----------
        alpha : TensorType[float]
            Probability of keeping an event.

        Returns:
        -------
        keep : Batch
            Batch of kept events.
        remove : Batch
            Batch of removed events.
        """
        if alpha.dim() == 1:
            keep = (
                torch.bernoulli(alpha[..., None].repeat(1, self.seq_len))[..., None]
                .repeat(1, 1, self.dim)
                .bool()
            )
        elif alpha.dim() == 2:
            keep = torch.bernoulli(alpha)[..., None].repeat(1, 1, self.dim).bool()
        else:
            raise Warning("alpha has too many dimensions")

        # remove from mask
        keep_mask = self.mask * keep
        rem_mask = self.mask * ~keep

        # shorten padding after removal
        return self.remove_unnescessary_padding(
            points=self.points * keep_mask,
            mask=keep_mask,
            kept=self.kept * keep_mask[:, :, 0] if self.kept is not None else self.kept,
            space_bound=self.space_bound,
        ), self.remove_unnescessary_padding(
            points=self.points * rem_mask,
            mask=rem_mask,
            kept=self.kept * rem_mask[:, :, 0] if self.kept is not None else self.kept,
            space_bound=self.space_bound,
        )

    def split_time(
        self,
        t_min: TensorType[float],
        t_max: TensorType[float],
    ) -> Tuple["Batch", "Batch", TensorType, TensorType]:
        """
        Time forecasting task from ADD-THIN.

        history_mask = self.time < t_min[:, None]
        forecast_mask = (self.time < t_max[:, None]) & ~history_mask

        # remove from mask
        forecast_mask = self.mask & forecast_mask
        history_mask = self.mask & history_mask

        # more than 5 events in history and more than one to be predicted
        batch_mask = (forecast_mask.sum(-1) > 1) & (history_mask.sum(-1) > 5)

        return (self.remove_unnescessary_padding(
                time=(self.time * history_mask)[batch_mask],
                mask=history_mask[batch_mask],
                kept=None,
                tmax=self.tmax,
            ),
            self.remove_unnescessary_padding(
                time=(self.time * forecast_mask)[batch_mask],
                mask=forecast_mask[batch_mask],
                kept=None,
                tmax=self.tmax,
            ),
            t_max[batch_mask],
            t_min[batch_mask])
        """
        history_mask = self.points[:, :, 0] < t_min[:, None]
        forecast_mask = (self.points[:, :, 0] < t_max[:, None]) & ~history_mask

        # remove from mask
        forecast_mask = self.mask & forecast_mask[:, :, None]
        history_mask = self.mask & history_mask[:, :, None]

        # more than 5 events in history and more than one to be predicted
        batch_mask = (forecast_mask[:, :, 0].sum(-1) > 1) & (
            history_mask[:, :, 0].sum(-1) > 5
        )

        # shorten padding after removal
        return (
            self.remove_unnescessary_padding(
                points=(self.points * history_mask)[batch_mask],
                mask=history_mask[batch_mask],
                kept=None,
                space_bound=self.space_bound,
            ),
            self.remove_unnescessary_padding(
                points=(self.points * forecast_mask)[batch_mask],
                mask=forecast_mask[batch_mask],
                kept=None,
                space_bound=self.space_bound,
            ),
            t_max[batch_mask],
            t_min[batch_mask],
        )

    def split_space(
        self,
        x_min: TensorType[float, "batch", "dim"],
        x_max: TensorType[float, "batch", "dim"],
    ) -> Tuple["Batch", "Batch", TensorType, TensorType]:
        """
        Split events according to space, taking events between [x_min, x_max]
        on all dimensions.

        Parameters:
        ----------
        x_min : TensorType[float, "batch", "dim"]
            Lower bound of space of events to keep.
        x_max : TensorType[float, "batch", "dim"]
            Upper bound of space of events to keep.

        Returns:
        -------
        condition : Batch
            Batch of events outside [x_min, x_max].
        forecast : Batch
            Batch of events inside [x_min, x_max].
        x_max : TensorType
            Upper bound of space of events to keep.
        x_min : TensorType
            Lower bound of space of events to keep.
        """
        assert x_min.dim() == 2, "time has too many or too low dimensions"
        assert x_max.dim() == 2, "time has too many or too low dimensions"

        # Create masks for points within the bounds
        forecast_mask = (
            ((self.points >= x_min[:, None, :]) & (self.points <= x_max[:, None, :]))
            .all(-1)[..., None]
            .expand_as(self.mask)
        )
        condition_mask = ~forecast_mask

        # remove from mask
        forecast_mask = self.mask & forecast_mask
        condition_mask = self.mask & condition_mask

        # shorten padding after removal
        return (
            self.remove_unnescessary_padding(
                points=(self.points * condition_mask),
                mask=condition_mask,
                kept=None,
                space_bound=self.space_bound,
            ),
            self.remove_unnescessary_padding(
                points=(self.points * forecast_mask),
                mask=forecast_mask,
                kept=None,
                space_bound=self.space_bound,
            ),
            x_max,
            x_min,
        )

    def split_space_func(self, func: callable, *args) -> Tuple["Batch", "Batch"]:
        forecast_mask = func(self.points[:, :, -2], self.points[:, :, -1], *args)[
            ..., None
        ].expand_as(self.mask)

        # Create masks for points within the bounds
        condition_mask = ~forecast_mask

        # remove from mask
        forecast_mask = self.mask & forecast_mask
        condition_mask = self.mask & condition_mask

        # shorten padding after removal
        return (
            self.remove_unnescessary_padding(
                points=(self.points * condition_mask),
                mask=condition_mask,
                kept=None,
                space_bound=self.space_bound,
            ),
            self.remove_unnescessary_padding(
                points=(self.points * forecast_mask),
                mask=forecast_mask,
                kept=None,
                space_bound=self.space_bound,
            ),
        )

    def _validate(self):
        """
        Validate batch, esp. masking.
        """
        # Check space bound
        # See if l_i < u_i for all dimensions
        assert (
            self.space_bound[:, 0] < self.space_bound[:, 1]
        ).all(), "wrong boundaries in space"

        # Check mask
        # mask as long as seq len;
        assert (self.mask.sum(1)[:, 0] == self.unpadded_length).all(), "wrong mask"
        assert (self.points * self.mask == self.points).all(), "wrong mask"
        # dimensions match
        assert self.mask.shape == (
            self.batch_size,
            self.seq_len,
            self.dim,
        ), f"mask has wrong shape {self.mask.shape}, expected {(self.batch_size, self.seq_len, self.dim)}"


@typechecked
def pad(sequences, length: Union[int, None] = None, value: float = 0):
    """
    Utility function to generate padding and mask for sequences.
    Parameters:
    ----------
            sequences: List of sequences.
            value: float = 0,
            length: Optional[int] = None,
    Returns:
    ----------
            sequences: Padded sequence,
                shape (batch_size, seq_length)
            mask: Boolean mask that indicates which entries
                do NOT correspond to padding, shape (batch_size, seq_len)
    """

    # Pad first sequence to enforce padding length
    if length:
        device = sequences[0].device
        dtype = sequences[0].dtype
        tensor_length = sequences[0].size(0)
        intial_pad = torch.empty(
            torch.Size([length]) + sequences[0].shape[1:],
            dtype=dtype,
            device=device,
        ).fill_(value)
        intial_pad[:tensor_length, ...] = sequences[0]
        sequences[0] = intial_pad

    sequences = pad_sequence(
        sequences, batch_first=True, padding_value=value
    )  # [order]

    return sequences


@typechecked
class SequenceDataset(torch.utils.data.Dataset):
    """Dataset of variable-length event sequences."""

    def __init__(
        self,
        sequences: List[Sequence],
    ):
        self.sequences = sequences
        self.space_bound = sequences[0].space_bound

    def __getitem__(self, idx: int):
        sequence = self.sequences[idx]
        return sequence

    def __len__(self) -> int:
        return len(self.sequences)

    def to(self, device: [torch.device, str]):  # type: ignore
        for sequence in self.sequences:
            sequence.to(device)


@typechecked
class DataModule(pl.LightningDataModule):
    """
    Datamodule for variable length event sequences for spatial point processes.

    Parameters:
    ----------
    root : str
        Path to data.
    name : str
        Name of dataset.
    data_type : str
        Type of data: spp or stpp
    split_seed : int
        Seed for random split.
    batch_size : int
        Batch size.
    train_size : float
        Percentage of data to use for training.
    val_size : float
        Percentage of data to use for validation.
    """

    def __init__(
        self,
        root: Path,
        name: str,
        data_type: str,
        split_seed: int = 80672983,
        batch_size: int = 32,
        train_size: float = 0.6,
        val_size: float = 0.2,
    ) -> None:
        super().__init__()
        self.root = root
        self.split_seed = split_seed
        self.batch_size = batch_size
        self.train_percentage = train_size
        self.val_percentage = val_size
        self.name = name
        self.data_type = data_type

        self.dataset = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_data(self) -> None:
        """Load sequence data from root."""
        file_name = (
            f"{self.name}_{self.data_type}"
            if not self.data_type == "tpp"
            else self.name
        )
        point_sequences, dict_keys = load_sequences(
            self.root, file_name, TPP=self.data_type == "tpp"
        )

        # Depending on the dataset the preparation differs
        if self.data_type == "tpp":
            self.dataset = SequenceDataset(sequences=point_sequences)
            self.space_bound = self.dataset.space_bound
            self.train_size = int(self.train_percentage * len(self.dataset))
            self.val_size = int(self.val_percentage * len(self.dataset))
            self.test_size = len(self.dataset) - (self.train_size + self.val_size)

            self.train_data, self.val_data, self.test_data = random_split(
                self.dataset,
                [self.train_size, self.val_size, self.test_size],
                generator=torch.Generator().manual_seed(self.split_seed),
            )

        if self.name == "hawkes":
            self.dataset = SequenceDataset(sequences=point_sequences)
            self.space_bound = self.dataset.space_bound
            self.train_size = int(self.train_percentage * len(self.dataset))
            self.val_size = int(self.val_percentage * len(self.dataset))
            self.test_size = len(self.dataset) - (self.train_size + self.val_size)

            self.train_data, self.val_data, self.test_data = random_split(
                self.dataset,
                [self.train_size, self.val_size, self.test_size],
                generator=torch.Generator().manual_seed(self.split_seed),
            )

        if "earthquakes":
            self.dataset = SequenceDataset(sequences=point_sequences)
            self.space_bound = self.dataset.space_bound
            # Then we do same split as Meta
            total_files = len(self.dataset)
            dataset_files = list(range(total_files))

            # Exclude files from training
            exclude_from_train = (
                dataset_files[::30]
                + dataset_files[1::30]
                + dataset_files[2::30]
                + dataset_files[3::30]
                + dataset_files[4::30]
                + dataset_files[5::30]
                + dataset_files[6::30]
                + dataset_files[7::30]
                + dataset_files[8::30]
                + dataset_files[9::30]
                + dataset_files[10::30]
            )

            val_files = dataset_files[3::30]
            test_files = dataset_files[7::30]
            train_files = list(set(dataset_files) - set(exclude_from_train))

            # Creating datasets
            self.train_data = torch.utils.data.Subset(self.dataset, train_files)
            self.val_data = torch.utils.data.Subset(self.dataset, val_files)
            self.test_data = torch.utils.data.Subset(self.dataset, test_files)

        if self.name == "citibike":
            # Create splitting patterns of Meta
            splits = {
                # Train from 2019-04 to 2019-07
                "train": lambda f: bool(re.match(r"20190[4567]\d\d_\d\d\d", f)),
                # Validation from 2019-08-01 to 2019-08-15
                "val": lambda f: bool(re.match(r"201908\d\d_\d\d\d", f))
                and int(re.match(r"201908(\d\d)_\d\d\d", f).group(1)) <= 15,
                # Validation from 2019-08-16 to 2019-08-31
                "test": lambda f: bool(re.match(r"201908\d\d_\d\d\d", f))
                and int(re.match(r"201908(\d\d)_\d\d\d", f).group(1)) > 15,
            }
            # Split the list of sequences according to the pattern
            train_sequences = [
                seq
                for key, seq in zip(dict_keys, point_sequences)
                if splits["train"](key)
            ]
            val_sequences = [
                seq
                for key, seq in zip(dict_keys, point_sequences)
                if splits["val"](key)
            ]
            test_sequences = [
                seq
                for key, seq in zip(dict_keys, point_sequences)
                if splits["test"](key)
            ]

            # Generate datasets
            self.dataset = SequenceDataset(point_sequences)
            self.train_data = SequenceDataset(train_sequences)
            self.val_data = SequenceDataset(val_sequences)
            self.test_data = SequenceDataset(test_sequences)

        if self.name == "covid_nj_cases":
            dates = dict()
            for k in dict_keys:  # save one unique key per day
                dates[k[:8]] = 1
            dates = list(dates.keys())

            # Reduce contamination between train/val/test splits.
            exclude_from_train = (
                dates[::27]
                + dates[1::27]
                + dates[2::27]
                + dates[3::27]
                + dates[4::27]
                + dates[5::27]
                + dates[6::27]
                + dates[7::27]
            )
            val_dates = dates[2::27]
            test_dates = dates[5::27]
            train_dates = list(set(dates) - set(exclude_from_train))

            # Generate indices
            train_sequences = [
                seq
                for key, seq in zip(dict_keys, point_sequences)
                if key[:8] in train_dates
            ]
            val_sequences = [
                seq
                for key, seq in zip(dict_keys, point_sequences)
                if key[:8] in val_dates
            ]
            test_sequences = [
                seq
                for key, seq in zip(dict_keys, point_sequences)
                if key[:8] in test_dates
            ]

            # Generate datasets
            self.dataset = SequenceDataset(point_sequences)
            self.train_data = SequenceDataset(train_sequences)
            self.val_data = SequenceDataset(val_sequences)
            self.test_data = SequenceDataset(test_sequences)

        if (
            self.name == "crime"
            or self.name == "pinwheel"
            or self.name == "independent"
        ):
            # Create splitting patterns of Meta
            splits = {
                # Train from 2019-04 to 2019-07
                "train": lambda f: bool(re.match(r"train(\d{4})", f)),
                # Validation from 2019-08-01 to 2019-08-15
                "val": lambda f: bool(re.match(r"val(\d{4})", f)),
                # Validation from 2019-08-16 to 2019-08-31
                "test": lambda f: bool(re.match(r"test(\d{4})", f)),
            }
            # Split the list of sequences according to the pattern
            train_sequences = [
                seq
                for key, seq in zip(dict_keys, point_sequences)
                if splits["train"](key)
            ]
            val_sequences = [
                seq
                for key, seq in zip(dict_keys, point_sequences)
                if splits["val"](key)
            ]
            test_sequences = [
                seq
                for key, seq in zip(dict_keys, point_sequences)
                if splits["test"](key)
            ]

            # Generate datasets
            self.dataset = SequenceDataset(point_sequences)
            self.train_data = SequenceDataset(train_sequences)
            self.val_data = SequenceDataset(val_sequences)
            self.test_data = SequenceDataset(test_sequences)

        self.get_statistics()

    def get_statistics(self):
        # Get train stats
        seq_lengths = []
        for i in range(len(self.train_data)):
            seq_lengths.append(len(self.train_data[i]))
        self.n_max = max(seq_lengths)
        self.n_mean = mean(seq_lengths)

    def setup(self, stage=None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            collate_fn=Batch.from_sequence_list,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=len(self.val_data),  # evaluate all at once
            collate_fn=Batch.from_sequence_list,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=len(self.test_data),  # evaluate all at once
            collate_fn=Batch.from_sequence_list,
            drop_last=False,
        )


def load_sequences(root, name: str, TPP=False) -> Tuple[List[Sequence], List[str]]:
    """Load dataset.

    Parameters:
    ----------
    root : str
        Path to data.
    name : str
        Name of dataset.

    Returns:
    -------
    point_sequences : List[Sequence]
        List of event sequences.
    keys: List[str]
        List of key names of sequences

    """

    if TPP:
        path = os.path.join(root, f"{name}.pkl")
        loader = torch.load(path, map_location=torch.device("cpu"))

        sequences = loader["sequences"]
        times = [seq["arrival_times"] for seq in sequences]
        tmax = loader["t_max"]

        space_bound = np.array([[0.0, tmax]])
        if not isinstance(space_bound, torch.Tensor):
            space_bound = torch.as_tensor(space_bound, dtype=torch.float32)

        # Re-scale batch points to [-1,1]^dim
        norm_space_bound = torch.cat(
            (
                torch.ones(space_bound.shape[0])[..., None] * (-1),
                torch.ones(space_bound.shape[0])[..., None] * (1),
            ),
            dim=-1,
        )
        point_sequences = []
        for idx in range(len(times)):
            points = times[idx]
            points = torch.as_tensor(points, dtype=torch.float32).unsqueeze(-1)
            # Order points on first dimension (if there is time, it is there)
            ind = points[:, 0].argsort(dim=0)
            points = points[ind]
            # Normalize point to [-1,1]
            points = (
                (points - space_bound[:, 0]) / (space_bound[:, 1] - space_bound[:, 0])
            ) * (norm_space_bound[:, 1] - norm_space_bound[:, 0]) + norm_space_bound[
                :, 0
            ]

            point_sequences.append(Sequence(points, space_bound=space_bound))
            dict_data = dict()
    else:
        path = os.path.join(root, f"{name}.npz")
        dict_data = np.load(path)

        # Get space boundaries
        space_bound = SPACE_BOUNDARIES[name]
        if not isinstance(space_bound, torch.Tensor):
            space_bound = torch.as_tensor(space_bound, dtype=torch.float32)

        # Re-scale batch points to [-1,1]^dim
        norm_space_bound = torch.cat(
            (
                torch.ones(space_bound.shape[0])[..., None] * (-1),
                torch.ones(space_bound.shape[0])[..., None] * (1),
            ),
            dim=-1,
        )

        point_sequences = []
        for key in dict_data.keys():
            points = dict_data[key]
            if not isinstance(points, torch.Tensor):
                points = torch.as_tensor(points, dtype=torch.float32)
            # Order points on first dimension (if there is time, it is there)
            ind = points[:, 0].argsort(dim=0)
            points = points[ind]
            # Normalize point to [-1,1]
            points = (
                (points - space_bound[:, 0]) / (space_bound[:, 1] - space_bound[:, 0])
            ) * (norm_space_bound[:, 1] - norm_space_bound[:, 0]) + norm_space_bound[
                :, 0
            ]

            point_sequences.append(Sequence(points, space_bound=space_bound))

    return point_sequences, list(dict_data.keys())
