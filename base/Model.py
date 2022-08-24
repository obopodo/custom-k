from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X, y):
        '''fit model'''
        ...
    
    
    @abstractmethod
    def predict(self, X):
        ...