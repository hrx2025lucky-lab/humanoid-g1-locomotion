"""Flat replay storage for adversarial policy samples."""

from __future__ import annotations

import torch


class ReplayBuffer:
    """A fixed-size sample reservoir with vectorized batch insertion."""

    def __init__(self, capacity: int, feature_dim: int, device: str | torch.device = "cpu") -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.device = torch.device(device)
        self._storage = torch.empty(self.capacity, self.feature_dim, device=self.device)
        self._write_index = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def data(self) -> torch.Tensor:
        """Return retained samples ordered from oldest to newest."""
        if self._size == 0:
            return self._storage[:0]
        start = (self._write_index - self._size) % self.capacity
        indices = (torch.arange(self._size, device=self.device) + start) % self.capacity
        return self._storage[indices]

    def append(self, samples: torch.Tensor) -> None:
        """Append a two-dimensional batch, evicting the oldest samples."""
        if samples.ndim != 2 or samples.shape[1] != self.feature_dim:
            actual = tuple(samples.shape)
            raise ValueError(
                f"Expected samples with feature dimension {self.feature_dim}, got shape {actual}."
            )
        samples = samples.detach().to(self.device)
        if samples.shape[0] >= self.capacity:
            self._storage.copy_(samples[-self.capacity :])
            self._write_index = 0
            self._size = self.capacity
            return

        count = samples.shape[0]
        first = min(count, self.capacity - self._write_index)
        self._storage[self._write_index : self._write_index + first].copy_(samples[:first])
        remainder = count - first
        if remainder:
            self._storage[:remainder].copy_(samples[first:])
        self._write_index = (self._write_index + count) % self.capacity
        self._size = min(self.capacity, self._size + count)

    def sample(self, batch_size: int) -> torch.Tensor:
        """Draw samples uniformly, with replacement only when necessary."""
        if self._size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size <= self._size:
            logical_indices = torch.randperm(self._size, device=self.device)[:batch_size]
        else:
            logical_indices = torch.randint(self._size, (batch_size,), device=self.device)
        oldest_index = (self._write_index - self._size) % self.capacity
        physical_indices = (logical_indices + oldest_index) % self.capacity
        return self._storage[physical_indices]
