import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU activation: f(x) = max(0, x)
    """
    x_relu = np.maximum(0,x)
    return x_relu