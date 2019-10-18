import numpy as np
import sprl.util.misc as util
from scipy.stats import multivariate_normal
from scipy.stats import norm
import matplotlib.patches as patches


class Gaussian:

    def __init__(self, lower_bounds, upper_bounds, mu, sigma):
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.mu = mu
        self.sigma = sigma

    def sample(self, n_samples=1):
        if n_samples == 1:
            return util.sample_normal_with_bounds(self.mu, self.sigma, self.lower_bounds, self.upper_bounds)
        else:
            samples = []
            for i in range(0, n_samples):
                samples.append(
                    util.sample_normal_with_bounds(self.mu, self.sigma, self.lower_bounds, self.upper_bounds))
            return np.array(samples)

    def get_bounds(self):
        return self.lower_bounds, self.upper_bounds

    def get_moments(self):
        return np.copy(self.mu), np.copy(self.sigma)

    def get_log_pdf(self, x):
        try:
            return multivariate_normal.logpdf(x, self.mu, self.sigma)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                "Encountered linalg error: " + str(e) + "\nMean: " + str(self.mu) + "\nSigma: " + str(self.sigma))

    @staticmethod
    def _eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    def init_visualization(self, ax, color, alpha=0.5, fill=False, dimensions=None):
        if self.mu.shape[0] == 1:
            sigma = np.squeeze(self.sigma)
            mu = np.squeeze(self.mu)
            x_s = np.linspace(mu - 3.5 * np.sqrt(sigma), mu + 3.5 * np.sqrt(sigma), 200)
            y_s = norm.pdf(x_s, mu, scale=np.sqrt(sigma))
            y_s /= np.max(y_s)

            if fill:
                line = ax.fill_between(x_s, y_s, np.zeros_like(y_s), alpha=alpha, color=color, lw=2)
            else:
                line, = ax.plot(x_s, y_s, lw=2, color=color)
            return line
        if self.mu.shape[0] == 2 or dimensions is not None:
            if dimensions is not None:
                l, v = self._eigsorted(self.sigma[np.ix_(dimensions, dimensions)])
            else:
                l, v = self._eigsorted(self.sigma)
            l = np.sqrt(l)
            theta = np.degrees(np.arctan2(*v[:, 0][::-1]))

            patch = patches.Ellipse(self.mu if dimensions is None else self.mu[dimensions], width=l[0] * 6,
                                    height=l[1] * 6, angle=theta, fill=fill, linewidth=2, color=color,
                                    alpha=alpha)

            ax.add_artist(patch)
            return patch
        else:
            return None

    def update_visualization(self, patch, dimensions=None):
        if self.mu.shape[0] == 1:
            # Update not implemented until now
            pass
        if self.mu.shape[0] == 2:
            if dimensions is not None:
                l, v = self._eigsorted(self.sigma[np.ix_(dimensions, dimensions)])
            else:
                l, v = self._eigsorted(self.sigma)
            l = np.sqrt(l)
            theta = np.degrees(np.arctan2(*v[:, 0][::-1]))
            if dimensions is not None:
                patch.center = self.mu[dimensions]
            else:
                patch.center = self.mu
            patch.width = l[0] * 6
            patch.height = l[1] * 6
            patch.angle = theta

    def refit(self, weights, xs):
        self.mu = np.sum(weights[:, None] * xs, axis=0) / np.sum(weights)
        diff = xs - self.mu[None, :]
        sum_w = np.sum(weights)
        v = sum_w / ((sum_w ** 2) - np.sum(weights ** 2))
        self.sigma = np.sum(weights[:, None, None] * np.einsum("ij, ik -> ijk", diff, diff), axis=0) / v
