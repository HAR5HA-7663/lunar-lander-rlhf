"""PyTorch Dataset for pairwise preference data."""

import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset


class PairwisePreferenceDataset(Dataset):
    """
    Loads saved pairwise preference data and returns tensors.

    Each item: (feat_A, feat_B, label)
        feat_A, feat_B: trajectory feature vectors (float32)
        label: 1.0 if A is preferred, 0.0 if B is preferred (float32)

    Expected pickle format: list of dicts with keys:
        'feat_A': np.ndarray, 'feat_B': np.ndarray, 'label': int (0 or 1)
    """

    def __init__(self, pairs_path: str | Path):
        pairs_path = Path(pairs_path)
        if not pairs_path.exists():
            raise FileNotFoundError(f"Preference pairs file not found: {pairs_path}")

        with open(pairs_path, "rb") as f:
            self.pairs = pickle.load(f)

        print(f"Loaded {len(self.pairs)} preference pairs from {pairs_path}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        feat_a = torch.tensor(pair["feat_A"], dtype=torch.float32)
        feat_b = torch.tensor(pair["feat_B"], dtype=torch.float32)
        label = torch.tensor(float(pair["label"]), dtype=torch.float32)
        return feat_a, feat_b, label
