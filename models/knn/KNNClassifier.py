import numpy as np
from .KNNAbstract import KNN


class KNNClassifier(KNN):
    def predict(self, X) -> np.ndarray:
        X = self._check_X(X)
        m, n = X.shape
        predicts = np.zeros((m,))
        for i, x in enumerate(X):
            neighbors_values = self._get_nn_values(x)
            predicts[i] = np.round(np.median(neighbors_values))

        return predicts


    def score(self, X, y) -> float:
        '''Returns accuracy'''
        X = self._check_X(X)
        y = np.asarray(y)
        preds = self.predict(X)
        return ( (y == preds).sum() ) / len(preds)
