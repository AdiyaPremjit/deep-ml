import numpy as np


def make_diagonal(x: np.ndarray) -> np.ndarray:
  return np.diag(x.astype(float))


# Example usage
x = np.array([1, 2, 3])
output = make_diagonal(x)
print(output)
