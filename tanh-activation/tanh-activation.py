import numpy as np

def tanh(x: list) -> np.ndarray:
    if isinstance(x, list):
        x = np.array(x)
    
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))