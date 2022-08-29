from abc import ABCMeta, abstractmethod
import numpy as np

class Model(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X, y):
        '''fit model'''
        self.predicts_ = np.zeros_like(y)
    
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        ...


    def fit_predict(self, X) -> np.ndarray:
        self.fit(X)
        return self.predicts_