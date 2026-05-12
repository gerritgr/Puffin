"""Dataset utilities for the AlphaEarth population disaggregation project."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class CoarseBlockDataset(Dataset):
    """PyTorch Dataset that yields one coarse block per item.

    Parameters
    ----------
    x_fine:
        Fine-cell feature array of shape ``(N, AE_DIM)``.
    y_coarse:
        Coarse-cell target array of shape ``(B,)`` where ``B`` is the
        number of coarse blocks.
    block_ids:
        Integer array of shape ``(N,)`` mapping each fine cell to its
        coarse block index in ``[0, B)``.
    block_list:
        Array of block indices to include in this dataset split.
    """

    def __init__(
        self,
        x_fine: np.ndarray,
        y_coarse: np.ndarray,
        block_ids: np.ndarray,
        block_list: np.ndarray,
    ) -> None:
        self.x_fine = torch.tensor(x_fine, dtype=torch.float32)
        self.y_coarse = torch.tensor(y_coarse, dtype=torch.float32)
        self.block_ids = torch.tensor(block_ids, dtype=torch.long)
        self.block_list = torch.tensor(block_list, dtype=torch.long)

        self.indices_by_block: list[Tensor] = []
        for b in self.block_list:
            idx = torch.where(self.block_ids == b)[0]
            self.indices_by_block.append(idx)

    def __len__(self) -> int:
        return len(self.block_list)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        block_id = self.block_list[idx]
        fine_idx = self.indices_by_block[idx]

        x_block = self.x_fine[fine_idx]
        y_block = self.y_coarse[block_id]

        return x_block, y_block, block_id


def collate_blocks(
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    """Collate a list of ``(x_block, y_block, block_id)`` tuples.

    All blocks are assumed to contain the same number of fine cells
    (``FINE_PER_COARSE``), so ``torch.stack`` is used for efficiency.
    If block sizes differ, use ``torch.nn.utils.rnn.pad_sequence`` instead.
    """
    x_blocks, y_blocks, block_ids_batch = zip(*batch)

    x_blocks_t = torch.stack(x_blocks, dim=0)
    y_blocks_t = torch.stack(y_blocks, dim=0)
    block_ids_t = torch.stack(block_ids_batch, dim=0)

    return x_blocks_t, y_blocks_t, block_ids_t
