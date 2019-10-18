import numpy as np
import scipy.optimize as opt
import sprl.util.misc as util


class CREPS:
    def __init__(self, feature_func, feature_mean, regularizer):
        self._feature_func = feature_func
        self._feature_mean = feature_mean
        self._last_eta = None
        self._last_phi = None
        self._regularizer = regularizer

    def reweight_samples(self, contexts, rewards, epsilon):
        self._minimize_joint_dual(rewards, self._feature_func(contexts), epsilon)

        weights = self._calculate_weights(self._feature_func(contexts), rewards)
        return weights

    def _minimize_joint_dual(self, rewards, contexts, epsilon):
        # We initialize the approximation of the Value Function using Ridge Regression to make the optimization
        # numerically more stable
        self._last_phi = util.ridge_regression(contexts, rewards, ridge_factor=1e-1)

        # We always start with a high value of eta (1000) because the optimization is numerically more instable
        # for small eta values (so we only should make the value small if we need that)
        res = opt.minimize(lambda x: self._dual_complete(x[0], x[1:], rewards, contexts, epsilon),
                           np.concatenate((np.array([1000.]), self._last_phi)), method="L-BFGS-B",
                           bounds=[(1e-10, None)] + [(None, None)] * contexts.shape[1],
                           jac=True, options={"maxiter": 100000, "maxfun": 100000})
        eta = res.x[0]
        phi = res.x[1:]

        self._last_eta = eta
        self._last_phi = phi

        last_cost, last_kl = self._dual(eta, phi, rewards, contexts, epsilon)
        print("Dual: " + ("%.4E" % last_cost) + ", KL: " + ("%.4E" % last_kl))

        return last_cost

    def _dual(self, eta, alpha, rewards, contexts, epsilon):
        n = rewards.shape[0]
        delta = rewards - np.dot(contexts, alpha)
        delta_offset = np.max(delta)

        delta_c = delta - delta_offset
        delta_c_eta = delta_c / eta
        exp_delta_c_eta = np.exp(delta_c_eta)
        sum_exp_delta_c_eta = np.sum(exp_delta_c_eta)
        mu_exp_delta_c_eta = sum_exp_delta_c_eta / n

        mask = exp_delta_c_eta >= 1e-300
        n_kl = np.sum(mask)
        sum_exp_kl = np.sum(exp_delta_c_eta[mask])

        return eta * epsilon + np.dot(self._feature_mean, alpha) + eta * np.log(mu_exp_delta_c_eta) + delta_offset, \
               (np.sum(exp_delta_c_eta[mask] * (delta[mask] / eta)) / sum_exp_kl) - \
               np.log(sum_exp_kl / n_kl) - delta_offset / eta

    def _dual_complete(self, eta, alpha, rewards, contexts, epsilon):
        dual, dual_eta = self._dual_eta(eta, alpha, rewards, contexts, epsilon)
        dual1, dual_alpha = self._dual_alpha(eta, alpha, rewards, contexts, epsilon)

        if np.abs(dual - dual1) > 1e-10:
            raise RuntimeError("Invalid computation of dual!")

        return dual, np.concatenate((dual_eta, dual_alpha))

    def _dual_eta(self, eta, alpha, rewards, contexts, epsilon):
        n = rewards.shape[0]
        delta = rewards - np.dot(contexts, alpha)
        delta_offset = np.max(delta)

        delta_c = delta - delta_offset
        delta_c_eta = delta_c / eta
        exp_delta_c_eta = np.exp(delta_c_eta)
        sum_exp_delta_c_eta = np.sum(exp_delta_c_eta)
        mu_exp_delta_c_eta = sum_exp_delta_c_eta / n

        penalty = self._regularizer * np.dot(alpha.T, alpha)
        return eta * epsilon + np.dot(self._feature_mean, alpha) + eta * np.log(mu_exp_delta_c_eta) + delta_offset \
               + penalty, \
               np.array([epsilon + (delta_offset / eta) + np.log(mu_exp_delta_c_eta) -
                         (np.sum(delta * exp_delta_c_eta) / (eta * sum_exp_delta_c_eta))])

    def _dual_alpha(self, eta, alpha, rewards, contexts, epsilon):
        n = rewards.shape[0]
        delta = rewards - np.dot(contexts, alpha)
        delta_offset = np.max(delta)

        delta_c = delta - delta_offset
        delta_c_eta = delta_c / eta
        exp_delta_c_eta = np.exp(delta_c_eta)
        sum_exp_delta_c_eta = np.sum(exp_delta_c_eta)
        mu_exp_delta_c_eta = sum_exp_delta_c_eta / n

        penalty = self._regularizer * np.dot(alpha.T, alpha)
        gradient_term = 2 * self._regularizer * alpha

        return eta * epsilon + np.dot(self._feature_mean, alpha) + eta * np.log(mu_exp_delta_c_eta) + delta_offset \
               + penalty, \
               self._feature_mean - (np.sum(contexts * exp_delta_c_eta[:, None], axis=0) / sum_exp_delta_c_eta) + \
               gradient_term

    def _calculate_weights(self, contexts, rewards):
        v = np.dot(contexts, self._last_phi)
        delta = rewards - v
        max_delta = np.max(delta)

        weights = np.exp(np.minimum((delta - max_delta) / self._last_eta, 500))

        # Add this for increased stability
        if np.all(weights < 1e-300):
            weights = np.ones_like(weights)

        return weights / np.sum(weights)
