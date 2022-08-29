from ..abstract import Model
import numpy as np
from numpy.linalg import norm

class KNN(Model):
    def __init__(self, k: int = 1) -> None:
        '''
        KNN model
        Parameters:
        -----------
        k: int, default=1
            Number of nearest neighbors
        '''
        self.k = k

    def fit(self, X, y) -> None:
        self._X = np.array(X)
        self._y = np.array(y)
        self.predicts_ = self._y

    def _get_nn_values(self, x: np.ndarray) -> np.ndarray:
        distances = norm(self._X - x, axis=1)
        # argpartition is faster than argsort
        min_indices = np.argpartition(distances, self.k)[:self.k]
        return self._y[min_indices]
