from typing import Iterable, Union
from ..abstract import Model
import numpy as np
from numpy.linalg import norm
# from sklearn.cluster import KMeans

class KMeans(Model):
    def __init__(
        self, 
        k_clusters: int, 
        *,
        init_means: Union[str, Iterable] = 'random',
        max_iter: int = 1000,
        epsilon: float = 1e-6,
        ) -> None:
        '''
        K-Means clustering
        Parameters:
        -----------
        k_clusters: int
            Number of desired clusters

        init_means: str or array-like, default='random'   

        '''
        self.k = k_clusters
        self.cluster_labels = np.arange(k_clusters)
        self._init_means = init_means
        self.max_iter = max_iter
        self.epsilon = epsilon


    def _coerse_init_means(self) -> np.ndarray:
        if isinstance(self._init_means, Iterable) and len(self._init_means) == self.k:
            return self._init_means
        raise NotImplementedError


    def predict(self, X) -> np.ndarray:
        dists = np.asarray([np.linalg.norm(X - mean, axis=1) for mean in self.means]).T
        predicts = np.argmin(dists, axis=1)
        return predicts


    def _update_means(self, X):
        new_means = np.asarray([
                np.mean(X[self.predicts_ == label], axis=0) 
                for label in self.cluster_labels
                ])
        return new_means
    
    
    @property
    def _is_converged(self) -> bool:
        '''Check if deviation from previous values is less than tolerance level'''
        return norm(self.means - self._prev_means) / norm(self._prev_means) < self.epsilon
        

    def fit(self, X: Iterable) -> None:
        X = np.asarray(X)
        self.nobs, self.ndim = X.shape
        self.means = self._coerse_init_means()
        self._prev_means = self.means

        for i in range(self.max_iter):
            self.predicts_ = self.predict(X)
            self.means = self._update_means(X)
            if self._is_converged:
                break
            self._prev_means = self.means

