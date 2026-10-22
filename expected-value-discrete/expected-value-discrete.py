import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    return float(sum(x[i] * p[i] for i in range(len(x))))