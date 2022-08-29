from .KNNAbstract import KNN
import numpy as np

class KNNRegressor(KNN):
    def __init__(self, k: int = 1) -> None:
        super().__init__(k)
        self.predict_vec = np.vectorize(self._pred_one)

    def _pred_one(self, x) -> np.ndarray:
        neighbors_values = self._get_nn_values(x)
        return np.mean(neighbors_values)


    def predict(self, X) -> np.ndarray:
        m, n = np.asarray(X).shape
        predicts = np.zeros((m,))
        for i, x in enumerate(X):
            neighbors_values = self._get_nn_values(x)
            predicts[i] = np.mean(neighbors_values)

        return predicts