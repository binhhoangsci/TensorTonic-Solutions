import numpy as np

def adam_step(param: list, grad: list, m: list, v: list, t: int, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    param = np.asarray(param, dtype = float)
    grad = np.asarray(grad, dtype = float)
    m = np.asarray(m, dtype = float)
    v = np.asarray(v, dtype = float)
    
    m_t = beta1 * m + (1 - beta1) * grad
    v_t = beta2 * v + (1 - beta2) * (grad ** 2)
    hatm_t = m_t / (1 - beta1 ** t)
    hatv_t = v_t / (1 - beta2 ** t)
    param = param - lr * hatm_t / (np.sqrt(hatv_t) + eps)

    return param, m_t, v_t