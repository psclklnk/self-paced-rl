import numpy as np
import scipy.optimize as opt
import sprl.util.misc as util


class SPRL:
    def __init__(self, feature_func, target_log_pdf, regularizer, min_dual=None):
        self._feature_func = feature_func
        self._target_log_pdf = target_log_pdf
        self._regularizer = regularizer
        self._min_dual = min_dual
        self._last_eta_p = None
        self._last_eta_mu = None
        self._last_phi = None

    def reweight_samples(self, contexts, cur_log_pdf_func, rewards, epsilon, alpha):
        cur_log_pdf = cur_log_pdf_func(contexts)
        target_log_pdf = self._target_log_pdf(contexts)
        context_features = self._feature_func(contexts)

        self._minimize_joint_dual(rewards, context_features, cur_log_pdf, target_log_pdf, epsilon, alpha)

        weights = self._calculate_weights(rewards, context_features, cur_log_pdf, target_log_pdf, alpha)

        return weights

    @staticmethod
    def _is_success(res):
        return res.success  # or res.message == b'ABNORMAL_TERMINATION_IN_LNSRCH'

    def _run_optimization(self, rewards, context_features, cur_log_pdf, target_log_pdf, epsilon, alpha,
                          use_gradients, reg):
        cost_fun = lambda x: self._dual_complete(x[0], x[1], x[2:], rewards, context_features, cur_log_pdf,
                                                 target_log_pdf, epsilon, alpha, reg, gradients=use_gradients)
        callback = None
        if not use_gradients:
            callback = lambda x: self._check_grad(x[0], x[1], x[2:], rewards, context_features,
                                                  cur_log_pdf, target_log_pdf, epsilon, alpha, lmdb=reg)

        res = opt.minimize(cost_fun, np.concatenate((np.array([1000., 1000.]), self._last_phi)), method="L-BFGS-B",
                           bounds=[(1e-10, None)] * 2 + [(None, None)] * context_features.shape[1],
                           jac=use_gradients, options={"maxiter": 100000, "maxfun": 100000}, callback=callback)

        return res

    def _minimize_joint_dual(self, rewards, context_features, cur_log_pdf, target_log_pdf, epsilon, alpha, rf=1e-1):
        self._last_phi = util.ridge_regression(context_features, rewards, ridge_factor=rf)
        res = self._run_optimization(rewards, context_features, cur_log_pdf, target_log_pdf, epsilon, alpha, True,
                                     self._regularizer)

        if not self._is_success(res) or (self._min_dual is not None and res.fun < self._min_dual):
            print("Optimization unsuccessful. Using gradient free method")
            res = self._run_optimization(rewards, context_features, cur_log_pdf, target_log_pdf, epsilon, alpha, False,
                                         self._regularizer)

        if not self._is_success(res) or (self._min_dual is not None and res.fun < self._min_dual):
            print("Optimization still unsuccessful. Using gradient free method with lmdb=" + str(rf))
            res = self._run_optimization(rewards, context_features, cur_log_pdf, target_log_pdf, epsilon, alpha, False,
                                         rf)

        if not self._is_success(res):
            raise RuntimeError("Optimization unsuccessful")
        else:
            self._success = True
            self._last_eta_p = res.x[0]
            self._last_eta_mu = res.x[1]
            self._last_phi = res.x[2:]

    def _dual_complete(self, eta_p, eta_mu, phi, rewards, contexts, cur_log_pdf, target_log_pdf, epsilon, alpha,
                       regularizer, gradients=False):
        n1 = rewards.shape[0]
        values = np.dot(contexts, phi)
        delta = rewards - values
        delta_offset = np.max(delta)

        delta_c = delta - delta_offset
        delta_c_eta = delta_c / eta_p
        exp_delta_c_eta = np.exp(delta_c_eta)
        sum_exp_delta_c_eta = np.sum(exp_delta_c_eta)
        mu_exp_delta_c_eta = sum_exp_delta_c_eta / n1

        values_mu = alpha * target_log_pdf - alpha * cur_log_pdf + values
        values_mu_offset = np.max(values_mu)
        values_mu_c = values_mu - values_mu_offset
        kappa = values_mu_c / (alpha + eta_mu)
        exp_context_weights = np.exp(kappa)
        sum_exp_context_weights = np.sum(exp_context_weights)
        mu_exp_context_weights = sum_exp_context_weights / n1

        eta_p_grad = np.array([epsilon + (delta_offset / eta_p) + np.log(mu_exp_delta_c_eta) -
                               (np.sum(delta * exp_delta_c_eta) / (eta_p * sum_exp_delta_c_eta))])
        eta_mu_grad = np.array([epsilon + (values_mu_offset / (alpha + eta_mu)) + np.log(mu_exp_context_weights) -
                                (np.sum(values_mu * exp_context_weights) / (
                                            (alpha + eta_mu) * sum_exp_context_weights))])

        alpha_grad = (np.sum(contexts * exp_context_weights[:, None], axis=0) / sum_exp_context_weights) \
                     - (np.sum(contexts * exp_delta_c_eta[:, None], axis=0) / sum_exp_delta_c_eta)

        alpha_grad += 2 * regularizer * phi
        penalty = regularizer * np.dot(phi.T, phi)

        if gradients:
            return eta_p * epsilon + eta_p * np.log(mu_exp_delta_c_eta) + delta_offset + \
                   eta_mu * epsilon + (alpha + eta_mu) * np.log(mu_exp_context_weights) + values_mu_offset + penalty, \
                   np.concatenate((eta_p_grad, eta_mu_grad, alpha_grad))
        else:
            return eta_p * epsilon + eta_p * np.log(mu_exp_delta_c_eta) + delta_offset + \
                   eta_mu * epsilon + (alpha + eta_mu) * np.log(mu_exp_context_weights) + values_mu_offset + penalty

    def _calculate_weights(self, rewards, contexts, cur_log_pdf, target_log_pdf, alpha):
        v = np.dot(contexts, self._last_phi)
        delta = rewards - v
        max_delta = np.max(delta)
        weights = np.exp((delta - max_delta) / self._last_eta_p)

        values_mu = alpha * target_log_pdf - alpha * cur_log_pdf + v
        values_mu_offset = np.max(values_mu)
        mu_weights = np.exp((values_mu - values_mu_offset) / (alpha + self._last_eta_mu))

        # Add this for increased stability
        if np.all(weights < 1e-300):
            print("Unstable policy weights - setting them to be uniform")
            weights = np.ones_like(weights)

        if np.all(mu_weights < 1e-300):
            print("Unstable sampling weights - setting them to be uniform")
            mu_weights = np.ones_like(mu_weights)

        # We normalize the weights so that they sum to 1
        return weights / np.sum(weights), mu_weights / np.sum(mu_weights)

    def _check_grad(self, eta_p, eta_mu, phi, rewards, contexts, cur_log_pdf, target_log_pdf, epsilon, alpha,
                    lmdb=None):
        x = np.concatenate((np.array([eta_p, eta_mu]), phi))
        cd = util.central_differences(lambda a:
                                      self._dual_complete(x[0], x[1], x[2:], rewards, contexts, cur_log_pdf,
                                                          target_log_pdf, epsilon, alpha, regularizer=lmdb),
                                      x)
        grad = self._dual_complete(eta_p, eta_mu, phi, rewards, contexts, cur_log_pdf, target_log_pdf,
                                   epsilon, alpha, gradients=True, regularizer=lmdb)[1]

        max_diff = np.max(np.abs(cd - grad) / np.maximum(cd, np.ones_like(cd)))
        if max_diff > 1e-2:
            print("Significant gradient difference: " + str(max_diff))
