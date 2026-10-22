import numpy as np

def sample_var_std(x: list) -> dict:
    if isinstance(x, list):
        x = np.array(x)

    sam_mean = np.mean(x)
    centered = x - sam_mean
    sam_var = float(1.0/(x.size - 1) * np.sum(centered ** 2))
    sam_std = float(np.sqrt(sam_var))
    return { "variance": sam_var, "standard_deviation": sam_std }