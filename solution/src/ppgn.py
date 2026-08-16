"""PPGN models and deterministic data utilities for the Revised-QM9 task."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

ATOMIC_NUMBERS = (1, 6, 7, 8, 9)
TARGET_NAMES = ("atomization", "homo", "lumo", "dipole")


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


@dataclass(frozen=True)
class SplitArrays:
    molecule_id: np.ndarray
    atomic_numbers: np.ndarray
    coordinates: np.ndarray
    atom_mask: np.ndarray
    targets: np.ndarray | None


def load_split(data_dir: Path, split: str) -> SplitArrays:
    archive = np.load(data_dir / f"{split}.npz", allow_pickle=False)
    return SplitArrays(
        molecule_id=archive["molecule_id"],
        atomic_numbers=archive["atomic_numbers"],
        coordinates=archive["coordinates"],
        atom_mask=archive["atom_mask"],
        targets=archive["targets"] if "targets" in archive.files else None,
    )


class MoleculeDataset(Dataset):
    def __init__(self, arrays: SplitArrays) -> None:
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays.molecule_id)

    def __getitem__(self, index: int):
        result = (
            torch.from_numpy(self.arrays.atomic_numbers[index]).long(),
            torch.from_numpy(self.arrays.coordinates[index]).float(),
            torch.from_numpy(self.arrays.atom_mask[index]).bool(),
        )
        if self.arrays.targets is None:
            return result
        return (*result, torch.from_numpy(self.arrays.targets[index]).float())


class PairTensorBuilder(nn.Module):
    """Create a documented dense order-2 molecular tensor."""

    def __init__(self) -> None:
        super().__init__()
        radii = torch.zeros(10, dtype=torch.float32)
        radii[1] = 0.31
        radii[6] = 0.76
        radii[7] = 0.71
        radii[8] = 0.66
        radii[9] = 0.57
        self.register_buffer("covalent_radii", radii)
        self.register_buffer("rbf_centers", torch.linspace(0.5, 4.0, 8))

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        coordinates: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, nodes = atomic_numbers.shape
        pair_mask = atom_mask.unsqueeze(2) & atom_mask.unsqueeze(1)
        eye = torch.eye(nodes, device=coordinates.device, dtype=torch.bool).unsqueeze(0)
        diagonal_mask = eye & pair_mask
        off_diagonal_mask = pair_mask & ~eye

        differences = coordinates.unsqueeze(2) - coordinates.unsqueeze(1)
        distances = torch.sqrt(
            torch.sum(differences * differences, dim=-1).clamp_min(1.0e-12)
        )
        radii = self.covalent_radii[atomic_numbers]
        cutoffs = 1.25 * (radii.unsqueeze(2) + radii.unsqueeze(1))
        bonds = (distances <= cutoffs) & off_diagonal_mask

        atom_features = torch.stack(
            [(atomic_numbers == value) for value in ATOMIC_NUMBERS], dim=-1
        ).to(coordinates.dtype)
        diagonal_features = atom_features.unsqueeze(2) * diagonal_mask.unsqueeze(-1).to(
            coordinates.dtype
        )

        safe_distances = distances.unsqueeze(-1)
        rbf = torch.exp(
            -2.0 * (safe_distances - self.rbf_centers.view(1, 1, 1, -1)) ** 2
        )
        rbf = rbf * off_diagonal_mask.unsqueeze(-1).to(coordinates.dtype)
        inverse_distance = torch.where(
            off_diagonal_mask,
            1.0 / distances.clamp_min(0.1),
            torch.zeros_like(distances),
        ).unsqueeze(-1)
        tensor = torch.cat(
            [
                diagonal_features,
                diagonal_mask.unsqueeze(-1).to(coordinates.dtype),
                bonds.unsqueeze(-1).to(coordinates.dtype),
                rbf,
                inverse_distance,
                pair_mask.unsqueeze(-1).to(coordinates.dtype),
            ],
            dim=-1,
        )
        return tensor, pair_mask


class PPGNBlock(nn.Module):
    def __init__(self, width: int, use_quadratic: bool) -> None:
        super().__init__()
        self.use_quadratic = use_quadratic
        self.entry_mlp = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        self.left = nn.Linear(width, width, bias=False)
        self.right = nn.Linear(width, width, bias=False)
        self.update = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, tensor: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.entry_mlp(tensor)
        if self.use_quadratic:
            left = self.left(hidden)
            right = self.right(hidden)
            quadratic = torch.einsum("bikc,bkjc->bijc", left, right)
            valid_nodes = (
                torch.diagonal(pair_mask, dim1=1, dim2=2).sum(dim=1).clamp_min(1)
            )
            quadratic = quadratic / valid_nodes[:, None, None, None]
        else:
            quadratic = torch.zeros_like(hidden)
        update = self.update(torch.cat([hidden, quadratic], dim=-1))
        result = self.norm(tensor + update)
        return result * pair_mask.unsqueeze(-1).to(result.dtype)


class MolecularPPGN(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int = 17,
        width: int = 48,
        blocks: int = 3,
        outputs: int = 4,
        use_quadratic: bool = True,
    ) -> None:
        super().__init__()
        self.builder = PairTensorBuilder()
        self.input_projection = nn.Linear(input_channels, width)
        self.blocks = nn.ModuleList(
            [PPGNBlock(width, use_quadratic) for _ in range(blocks)]
        )
        self.readout = nn.Sequential(
            nn.Linear(width * 2, width * 2),
            nn.SiLU(),
            nn.Linear(width * 2, outputs),
        )

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        coordinates: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        tensor, pair_mask = self.builder(atomic_numbers, coordinates, atom_mask)
        hidden = self.input_projection(tensor)
        hidden = hidden * pair_mask.unsqueeze(-1).to(hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, pair_mask)

        nodes = atom_mask.sum(dim=1).clamp_min(1).to(hidden.dtype)
        diagonal = torch.diagonal(hidden, dim1=1, dim2=2).transpose(1, 2)
        diagonal_sum = diagonal.sum(dim=1) / nodes[:, None]
        total_sum = hidden.sum(dim=(1, 2))
        off_diagonal_sum = total_sum - diagonal.sum(dim=1)
        pairs = (nodes * (nodes - 1.0)).clamp_min(1.0)
        off_diagonal_mean = off_diagonal_sum / pairs[:, None]
        return self.readout(torch.cat([diagonal_sum, off_diagonal_mean], dim=-1))


class RawUnitRegressor(nn.Module):
    def __init__(
        self,
        model: MolecularPPGN,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        coordinates: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        standardized = self.model(atomic_numbers, coordinates, atom_mask)
        return standardized * self.target_std + self.target_mean


class ExpressivityPPGN(nn.Module):
    def __init__(
        self,
        *,
        width: int = 32,
        blocks: int = 2,
        use_quadratic: bool = True,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(3, width)
        self.blocks = nn.ModuleList(
            [PPGNBlock(width, use_quadratic) for _ in range(blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.SiLU(),
            nn.Linear(width, 2),
        )

    def forward(self, adjacency: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        _, nodes, _ = adjacency.shape
        pair_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        eye = torch.eye(nodes, device=adjacency.device, dtype=torch.bool).unsqueeze(0)
        diagonal = (eye & pair_mask).to(adjacency.dtype)
        off_diagonal = (pair_mask & ~eye).to(adjacency.dtype)
        tensor = torch.stack([diagonal, adjacency, off_diagonal], dim=-1)
        hidden = self.input_projection(tensor)
        hidden = hidden * pair_mask.unsqueeze(-1).to(hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, pair_mask)

        valid_nodes = node_mask.sum(dim=1).clamp_min(1).to(hidden.dtype)
        diagonal_hidden = torch.diagonal(hidden, dim1=1, dim2=2).transpose(1, 2)
        diagonal_mean = diagonal_hidden.sum(dim=1) / valid_nodes[:, None]
        off_sum = hidden.sum(dim=(1, 2)) - diagonal_hidden.sum(dim=1)
        pair_count = (valid_nodes * (valid_nodes - 1.0)).clamp_min(1.0)
        off_mean = off_sum / pair_count[:, None]
        return self.head(torch.cat([diagonal_mean, off_mean], dim=-1))


def load_manifest(data_dir: Path) -> dict:
    return json.loads((data_dir / "dataset_manifest.json").read_text())
