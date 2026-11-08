from math import log

def log_loss(y_true: list, y_pred: list, eps: float = 1e-15) -> list:
    loss = []
    n = len(y_true)
    for i in range(n):
        if y_pred[i] < eps:
            y_pred[i] = eps
        elif y_pred[i] > 1 - eps:
            y_pred[i] = 1 - eps
        loss.append(float(-(y_true[i] * log(y_pred[i]) + (1 - y_true[i]) * log(1 - y_pred[i]))))

    return loss