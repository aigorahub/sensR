from dataclasses import dataclass
from scipy.stats import betabinom
import numpy as np

__all__ = ["BetaBinomial"]

@dataclass
class BetaBinomial:
    """Simple beta-binomial model."""
    alpha: float
    beta: float
    n: int

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def pmf(self, k: int) -> float:
        return betabinom.pmf(k, self.n, self.alpha, self.beta)

    def logpmf(self, k: int) -> float:
        return betabinom.logpmf(k, self.n, self.alpha, self.beta)
