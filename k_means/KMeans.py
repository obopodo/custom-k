from typing import Iterable, Union
from ..base import Model
import numpy as np

# from sklearn.cluster import KMeans

class KMeans(Model):
    def __init__(
        self, 
        k_clusters: int, 
        *,
        init_means: Union[str, Iterable] = 'random'
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


    def _coerse_init_means(self) -> np.ndarray:
        raise NotImplementedError


    def fit(self, X: Iterable) -> None:
        X = np.asarray(X)
        m, self.ndim = X.shape
        self.means = self._coerse_init_means()
        self.predicts_ = np.zeros((m, ))
        raise NotImplementedError
