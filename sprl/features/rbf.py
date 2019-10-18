import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import functools
import operator


class RadialBasisFunctions:

    def __init__(self, n_features, intervals, kernel_widths=None, bias=False, seed=None, dim_mask=None,
                 width_multiplier=1.2):
        self._seed = seed
        self._n_features = n_features
        self._original_kernel_widths = kernel_widths
        self._centers, self._kernel_widths = self._compute_centers(self._seed, self._n_features, intervals,
                                                                   self._original_kernel_widths, width_multiplier)
        self._bias = bias

        if dim_mask is not None:
            self._dim_mask = dim_mask
        else:
            self._dim_mask = np.ones(self._centers.shape[2], dtype=bool)

    @staticmethod
    def _compute_centers(seed, n_features, intervals, kernel_widths, width_multiplier):
        if seed is not None:
            rand.seed(seed)

        if isinstance(n_features, (list, tuple)):
            n_dimensions = len(n_features)
            n_features_dim = n_features
            features_diff = 0
        else:
            n_dimensions = len(intervals)
            n_features_dim = int(np.float_power(n_features, 1.0 / float(n_dimensions)))
            features_diff = n_features - n_features_dim ** n_dimensions
            n_features_dim = [n_features_dim] * n_dimensions

        xds = np.meshgrid(
            *[np.linspace(interval[0], interval[1], f + 2)[1:-1] for f, interval in zip(n_features_dim, intervals)])
        centers = np.concatenate([np.expand_dims(xd, axis=n_dimensions) for xd in xds], axis=n_dimensions)
        centers = np.reshape(centers, (functools.reduce(operator.mul, centers.shape[0:-1], 1), centers.shape[-1]))

        random_centers = np.zeros((features_diff, n_dimensions))
        for i in range(0, n_dimensions):
            random_centers[:, i] = rand.uniform(intervals[i][0], intervals[i][1], features_diff)

        if kernel_widths is None:
            kernel_widths = []
            for i in range(0, n_dimensions):
                kernel_widths.append((intervals[i][1] - intervals[i][0]) * (width_multiplier / (2 * n_features_dim[i])))
            kernel_widths = np.array(kernel_widths)

        return np.expand_dims(np.concatenate((centers, random_centers), axis=0), axis=0), kernel_widths

    def __call__(self, x):
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)

        x = np.expand_dims(x, axis=1)
        x = x[:, :, self._dim_mask]
        diffs = x - self._centers
        activations = np.exp(-np.einsum('ijk,ijk -> ij', diffs * self._kernel_widths, diffs))
        activations = activations / np.sum(activations, axis=1)[:, None]
        if self._bias:
            activations = np.concatenate((activations, np.ones((activations.shape[0], 1))), axis=1)
        return activations

    def visualize(self):
        min_c = np.squeeze(np.min(self._centers, axis=1))
        max_c = np.squeeze(np.max(self._centers, axis=1))
        if self._centers.shape[2] == 1:
            xs = np.linspace(min_c, max_c, 200)[:, None]
            activations = self(xs)
            if self._bias:
                activations = activations[:, 0:-1]

            plt.plot(np.squeeze(xs), np.squeeze(activations))
        elif self._centers.shape[2] == 2:
            xs = np.linspace(min_c[0], max_c[0], 200)
            ys = np.linspace(min_c[1], max_c[1], 200)

            mxs, mys = np.meshgrid(xs, ys)
            activations = np.reshape(self(np.reshape(np.concatenate((mxs[:, :, None], mys[:, :, None]), axis=2),
                                                     (200 * 200, -1))),
                                     (200, 200, -1))
            if self._bias:
                activations = activations[:, :, 0:-1]

            activations = np.max(activations, axis=2)
            f = plt.figure()
            ax = f.gca()
            c = ax.pcolormesh(mxs, mys, activations, cmap='RdBu', vmin=np.min(activations), vmax=np.max(activations))
            ax.scatter(self._centers[0, :, 0], self._centers[0, :, 1])
            plt.colorbar(c, ax=ax)
        else:
            raise RuntimeError("Cannot visualize more than 2 dimensions")
        plt.show()
