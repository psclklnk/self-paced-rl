import sklearn.preprocessing as pre
import numpy as np


class PolynomialFeatures:

    def __init__(self, order, bias=True):
        self._f = pre.PolynomialFeatures(order, include_bias=bias)

    def __call__(self, x):
        if len(x.shape) == 1:
            return self._f.fit_transform(np.array([x]))
        elif len(x.shape) == 2:
            return self._f.fit_transform(x)
        else:
            res = []
            for i in range(0, x.shape[0]):
                res.append(self._f.fit_transform(x[i, :, :]))
            return np.array(res)
