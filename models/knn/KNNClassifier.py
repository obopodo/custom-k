from .KNNAbstract import KNN
import numpy as np

class KNNClassifier(KNN):
    def predict(self, X) -> np.ndarray:
        m, n = np.asarray(X).shape
        predicts = np.zeros((m,))
        for i, x in enumerate(X):
            neighbors_values = self._get_nn_values(x)
            predicts[i] = np.round(np.median(neighbors_values))

        return predicts
