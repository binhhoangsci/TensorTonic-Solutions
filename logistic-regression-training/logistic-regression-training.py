import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    N, d = X.shape
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    
    w = np.zeros(d)
    b = 0.0
    for _ in range(steps):
        hat_y = _sigmoid(X @ w + b)
        w -= lr/N * X.T @ (hat_y - y)
        b -= lr * np.mean(hat_y - y)

    return w, b