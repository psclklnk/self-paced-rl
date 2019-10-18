import numpy as np
import sprl.util.misc as util


class KLWLRPolicy:

    def __init__(self, lower_bounds, upper_bounds, mu_init, sigma_init, feature_func, epsilon, max_eta=100.):
        self._mu = mu_init

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        self._epsilon = epsilon
        self._max_eta = max_eta
        self._feature_func = feature_func
        self._theta = None
        self._sigma = sigma_init

    def compute_greedy_action(self, state):
        if self._theta is not None:
            features = np.squeeze(self._feature_func(state))
            mu = np.dot(features, self._theta)

            return mu
        else:
            return self._mu

    def compute_variance(self, state):
        return self._sigma

    def sample_action(self, state):
        mu = self.compute_greedy_action(state)
        sigma = self.compute_variance(state)

        return util.sample_normal_with_bounds(mu, sigma, self.lower_bounds, self.upper_bounds)

    @staticmethod
    def _kl_divergence(eta, weights, features, ys, mus_y, sigma_y):
        theta_new, sigma_y_new = util.new_gaussian_pol_eta(weights, features, ys, mus_y, sigma_y, eta)

        mus_new = np.dot(features, theta_new)
        kl_divs_y = util.gaussian_kl_divergences(mus_y, sigma_y, mus_new, sigma_y_new)

        return np.mean(kl_divs_y)

    def refit(self, weights, states, actions):
        features = self._feature_func(states)
        mus_y_cur = np.repeat(self._mu[None, :], weights.shape[0],
                              axis=0) if self._theta is None else np.dot(features, self._theta)

        eta_opt = util.ensure_kl_divergence(
            lambda eta: KLWLRPolicy._kl_divergence(eta, weights, features, actions, mus_y_cur, self._sigma),
            self._epsilon, max_eta=self._max_eta)

        self._theta, self._sigma = util.new_gaussian_pol_eta(weights, features, actions, mus_y_cur,
                                                             self._sigma, eta_opt)
