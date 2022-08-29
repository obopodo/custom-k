from abc import ABCMeta, abstractmethod
import numpy as np
from numpy.linalg import norm

from ..abstract import Model


class KNN(Model, metaclass=ABCMeta):
    def __init__(self, k_neighbors: int = 1) -> None:
        '''
        Abstract class for KNN model
        Parameters:
        -----------
        k_neighbors: int, default=1
            Number of nearest neighbors
        '''
        self.k_neighbors = k_neighbors


    def fit(self, X, y) -> 'KNN':
        self._X = self._check_X(X)
        self._y = np.asarray(y)
        self.predicts_ = self._y
        self.n_features_in_ = self._X.shape[1]
        return self


    def _get_nn_values(self, x: np.ndarray) -> np.ndarray:
        distances = norm(self._X - x, axis=1)
        # argpartition is faster than argsort
        min_indices = np.argpartition(distances, self.k_neighbors)[:self.k_neighbors]
        return self._y[min_indices]
