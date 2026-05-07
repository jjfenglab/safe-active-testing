import numpy as np

def to_safe_prob(pred, eps=1e-7):
    return np.clip(pred, eps, 1 - eps)