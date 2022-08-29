from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator


class Model(BaseEstimator, metaclass=ABCMeta):
    def _check_X(self, X) -> np.ndarray:
        X = np.asarray(X)
        if len(X.shape) == 1:
            raise ValueError(
                'Expected 2D array, got 1D array instead. '
                'Reshape your data either using array.reshape(-1, 1) '
                'if your data has a single feature '
                'or array.reshape(1, -1) if it contains a single sample.'
                )
        if len(X.shape) > 2:
            raise ValueError(f'Expected 2D array, got {len(X.shape)} array instead.')

        return X


    @abstractmethod
    def fit(self, X, y) -> 'Model':
        '''fit model'''
        self._check_X(X)
    
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        self._check_X(X)


    def fit_predict(self, X, y) -> np.ndarray:
        self.fit(X, y)
        return self.predicts_


    @abstractmethod
    def score(self, X, y) -> float:
        ...
