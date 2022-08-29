import numpy as np
from .KNNAbstract import KNN


class KNNRegressor(KNN):
    def predict(self, X) -> np.ndarray:
        X = self._check_X(X)
        m, n = X.shape
        predicts = np.zeros((m,))
        for i, x in enumerate(X):
            neighbors_values = self._get_nn_values(x)
            predicts[i] = np.mean(neighbors_values)

        return predicts


    def score(self, X, y) -> float:
        '''Returns coefficient of determination'''
        X = self._check_X(X)
        y = np.asarray(y)
        
        preds = self.predict(X)
        R2 = 1 - ( (y - preds)**2 ).sum() / ( (y - y.mean())**2 ).sum()
        return R2
