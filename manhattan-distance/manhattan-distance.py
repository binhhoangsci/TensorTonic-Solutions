import numpy as np

def manhattan_distance(x: list, y: list) -> float:
    x = np.array(x)
    y = np.array(y)
    n = x.size
    z = abs(x - y)
    return float(sum(z[i] for i in range(n)))