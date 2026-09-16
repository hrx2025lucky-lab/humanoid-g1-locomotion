"""Small task-independent controls for stable adversarial motion-prior training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiscriminatorBalanceController:
    """Choose a discriminator budget from its current policy/expert score separation."""

    max_updates: int = 4
    saturated_updates: int = 0
    saturation_threshold: float = 0.8
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.max_updates < 0:
            raise ValueError("max_updates must be non-negative.")
        if not 0 <= self.saturated_updates <= self.max_updates:
            raise ValueError("saturated_updates must be between zero and max_updates.")
        if not 0.0 < self.saturation_threshold <= 1.0:
            raise ValueError("saturation_threshold must be in (0, 1].")

    def recommended_updates(self, policy_score: float, expert_score: float) -> int:
        """Pause updates while the classifier already separates both classes."""
        if not self.enabled:
            return self.max_updates
        saturated = (
            policy_score <= -self.saturation_threshold
            and expert_score >= self.saturation_threshold
        )
        return self.saturated_updates if saturated else self.max_updates
