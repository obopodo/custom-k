from abc import ABCMeta, abstractmethod
import numpy as np

class Model(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X, y):
        '''fit model'''
        ...
    
    
    @abstractmethod
    def predict(self, X):
        ...


    def fit_predict(self, X) -> np.ndarray:
        self.fit(X)
        return self.predicts_