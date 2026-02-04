from dataclasses import dataclass
import numpy as np


@dataclass
class LinearConstraint:
    G: np.ndarray  # Matrix G where G * u <= b
    b: np.ndarray  # Vector b
    hard: bool = True
    priority: int = (
        0  # Higher value means higher priority (if we were to resolve conflicts)
    )
