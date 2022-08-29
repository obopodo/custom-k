from typing import Any, Iterable, Union
import numpy as np
from numpy.linalg import norm

from ..abstract import Model


class KMeans(Model):
    def __init__(
        self, 
        k_clusters: int = 2, 
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
            Must be array of shape (k_clusters, n_features) or "random"

        '''
        self.k_clusters = k_clusters
        self.init_means = init_means
        self.max_iter = max_iter
        self.epsilon = epsilon        


    def predict(self, X) -> np.ndarray:
        X = self._check_X(X)
        dists = np.asarray([norm(X - mean, axis=1) for mean in self.means_]).T
        predicts = np.argmin(dists, axis=1)
        return predicts


    def _coerse_init_means(self, X: np.ndarray) -> np.ndarray:
        if isinstance(self.init_means, str):
            if self.init_means == 'random':
                indices = np.random.choice(self.nobs_, self.k_clusters)
                return X[indices]
        elif isinstance(self.init_means, Iterable):
            self.init_means = np.asarray(self.init_means)
            if self.init_means.shape != (self.k_clusters, self.ndim_):
                raise ValueError(f'`init_means` must have shape ({self.k_clusters}, {self.ndim_}), got {self.init_means.shape}')
            return self.init_means
        raise ValueError(f'`init_means` must be array of shape ({self.k_clusters}, {self.ndim_}) or "random"')


    def _update_means(self, X):
        new_means = np.asarray([
                np.mean(X[self.predicts_ == label], axis=0) 
                for label in self.cluster_labels_
                ])
        return new_means
    
    
    @property
    def _is_converged(self) -> bool:
        '''Check if deviation from previous values is less than tolerance level'''
        return norm(self.means_ - self._prev_means) / norm(self._prev_means) < self.epsilon
        

    def fit(self, X: Iterable, y: Any = None) -> 'KMeans':
        '''
        Perform k-means clustering
        Parameters:
        -----------
        X: array-like
            Training dataset
        y : Ignored
            Not used, present here for API consistency by convention.
        '''
        X = self._check_X(X)
        self.cluster_labels_ = np.arange(self.k_clusters)
        self.nobs_, self.ndim_ = X.shape
        self.means_ = self._coerse_init_means(X)
        self._prev_means = self.means_

        for _ in range(self.max_iter):
            self.predicts_ = self.predict(X)
            self.means_ = self._update_means(X)
            if self._is_converged:
                break
            self._prev_means = self.means_
        
        return self


    def score(self, X: Iterable, y: Any = None) -> float:
        '''
        Compute negative objective of k-means algorithm 
        (sum of squared distances to cluster centres)

        Parameters:
        -----------
        X: array-like
            Dataset
        y : Ignored
            Not used, present here for API consistency by convention.
    '''
        X = np.asarray(X)
        preds = self.predict(X)
        score = 0
        for c in self.cluster_labels_:
            score -= ( (X[preds==c] - self.means_[c])**2 ).sum()
        return score
