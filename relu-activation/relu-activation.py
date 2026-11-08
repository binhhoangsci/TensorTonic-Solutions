import numpy as np

def relu(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x > 0.0, x, 0.0)