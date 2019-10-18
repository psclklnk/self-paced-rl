import numpy as np
import sprl.util.misc as util
from scipy.stats import multivariate_normal
from scipy.stats import norm
import matplotlib.patches as patches


class KLGaussian:

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
        elif self.mu.shape[0] == 2 or dimensions is not None:
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
        if self.mu.shape[0] == 2 or dimensions is not None:
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


class KLPolicy:

    def __init__(self, lower_bounds, upper_bounds, mu_init, sigma_init, feature_func):
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.feature_func = feature_func
        self.theta = None
        self.mu = mu_init
        self.sigma = sigma_init

    def compute_greedy_action(self, state):
        if self.theta is not None:
            features = np.squeeze(self.feature_func(state))
            return np.dot(features, self.theta)
        else:
            return self.mu

    def compute_variance(self, state):
        return self.sigma

    def sample_action(self, state):
        mu = self.compute_greedy_action(state)
        sigma = self.compute_variance(state)

        return util.sample_normal_with_bounds(mu, sigma, self.lower_bounds, self.upper_bounds)


class KLJoint:

    def __init__(self, lower_bounds_x, upper_bounds_x, mu_x, sigma_x, lower_bounds_y, upper_bounds_y, mu_y, sigma_y,
                 feature_func, epsilon, max_eta=100.):
        self.distribution = KLGaussian(lower_bounds_x, upper_bounds_x, mu_x, sigma_x)
        self.policy = KLPolicy(lower_bounds_y, upper_bounds_y, mu_y, sigma_y, feature_func)

        self.epsilon = epsilon
        self.max_eta = max_eta

    @staticmethod
    def _kl_divergence(eta, weights, xs, features, ys, mu_x, sigma_x, mus_y, sigma_y, x_weights=None):
        mu_x_new, sigma_x_new = util.new_gaussian_dist_eta(weights if x_weights is None else x_weights, xs, mu_x,
                                                           sigma_x, eta)
        theta_new, sigma_y_new = util.new_gaussian_pol_eta(weights, features, ys, mus_y, sigma_y, eta)

        mus_new = np.dot(features, theta_new)
        kl_div_x = util.gaussian_kl_divergences(mu_x[None, :], sigma_x, mu_x_new[None, :], sigma_x_new)[0]
        kl_divs_y = util.gaussian_kl_divergences(mus_y, sigma_y, mus_new, sigma_y_new)

        return np.mean(kl_divs_y) + kl_div_x

    def refit(self, weights, xs, ys, x_weights=None):
        features = self.policy.feature_func(xs)
        mus_y_cur = np.repeat(self.policy.mu[None, :], weights.shape[0],
                              axis=0) if self.policy.theta is None else np.dot(features, self.policy.theta)

        eta_opt = util.ensure_kl_divergence(
            lambda eta: KLJoint._kl_divergence(eta, weights, xs, features, ys, self.distribution.mu,
                                               self.distribution.sigma,
                                               mus_y_cur, self.policy.sigma, x_weights=x_weights), self.epsilon,
            max_eta=self.max_eta)

        mu_x, sigma_x = util.new_gaussian_dist_eta(weights if x_weights is None else x_weights, xs,
                                                   self.distribution.mu, self.distribution.sigma, eta_opt)
        theta, sigma_y = util.new_gaussian_pol_eta(weights, features, ys, mus_y_cur, self.policy.sigma, eta_opt)

        self.distribution.mu = mu_x
        self.distribution.sigma = sigma_x

        self.policy.theta = theta
        self.policy.sigma = sigma_y
