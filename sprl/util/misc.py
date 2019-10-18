import numpy as np
import pickle
import os
import numpy.random as rand
import scipy.optimize as opt
import scipy.linalg as scpla


def new_gaussian_pol_eta(weights, features, ys, mus_y, sigma_y, eta, lmbd=1e-10):
    w = np.sum(weights)
    n = weights.shape[0]

    # Compute the new mean
    reg = lmbd * np.eye(features.shape[1])
    tmp = np.sum((weights + eta / n)[:, None, None] * np.einsum("ki,kj->kij", features, features), axis=0) + reg
    tmp1 = np.sum(np.einsum("ki,kj->kij", features, weights[:, None] * ys + (eta / n) * mus_y), axis=0)
    theta_new = np.linalg.solve(tmp, tmp1)

    mus_y_new = np.dot(features, theta_new)
    diffs = ys - mus_y_new
    sigma_s = np.sum(weights[:, None, None] * np.einsum("ij,ik->ijk", diffs, diffs), axis=0)
    mu_diff = mus_y_new - mus_y
    sigma_s1 = np.sum(np.einsum("ij,ik->ijk", mu_diff, mu_diff), axis=0) / n
    sigma_y_new = (sigma_s + eta * (sigma_y + sigma_s1)) / (w + eta)

    return theta_new, sigma_y_new


def new_gaussian_dist_eta(weights, xs, mu_x, sigma_x, eta):
    w = np.sum(weights)

    mu_x_new = (np.sum(weights[:, None] * xs, axis=0) + eta * mu_x) / (w + eta)
    diffs = xs - mu_x_new[None, :]
    sigma_s = np.sum(weights[:, None, None] * np.einsum("ij,ik->ijk", diffs, diffs), axis=0)
    mu_diff = mu_x - mu_x_new
    sigma_x_new = (sigma_s + eta * (sigma_x + np.einsum("i,j->ij", mu_diff, mu_diff))) / (w + eta)

    return mu_x_new, sigma_x_new


def gaussian_kl_divergences(mus_1, sigma_1, mus_2, sigma_2):
    l = np.linalg.cholesky(np.tril(sigma_2, -1) + np.tril(sigma_2).T + np.eye(sigma_2.shape[0]) * 1e-10)
    tr = np.trace(scpla.solve_triangular(l, scpla.solve_triangular(l, sigma_1, trans=0, lower=True),
                                         trans=1, lower=True))
    mu_diff = mus_2 - mus_1
    mh_diffs = np.sum(
        mu_diff * scpla.solve_triangular(l, scpla.solve_triangular(l, mu_diff.T, trans=0, lower=True),
                                         trans=1, lower=True).T, axis=1)

    s_1, ld_1 = np.linalg.slogdet(sigma_1)
    s_2, ld_2 = np.linalg.slogdet(sigma_2)
    return 0.5 * (tr + mh_diffs - mus_1.shape[1] + s_2 * ld_2 - s_1 * ld_1)


def gaussian_log_likelihoods(actions, mus, sigma):
    s, ld = np.linalg.slogdet(sigma)
    diffs = actions - mus
    return -0.5 * (s * ld + np.sum(diffs * np.linalg.solve(sigma, diffs.T).T, axis=1) +
                   mus.shape[1] * np.log(2 * np.pi))


def ensure_kl_divergence(kl, max_kl, max_eta):
    # We only regularize the solution if necessary
    try:
        if kl(0) > max_kl:
            init_min_eta = 0.
            init_max_eta = 1.
            abort = False
            while kl(init_max_eta) > max_kl:
                if init_max_eta >= max_eta:
                    abort = True
                    break

                init_min_eta = init_max_eta
                init_max_eta = np.minimum(max_eta, init_max_eta * 10.)

            if abort:
                print("Could not regularize the problem to satisfaction, will use eta=" + str(max_eta))
                eta_opt = max_eta
            else:
                xtol = 5e-3
                rtol = 4 * np.finfo(float).eps
                eta_opt = opt.bisect(lambda eta: kl(eta) - max_kl, init_min_eta, init_max_eta, xtol=xtol, rtol=rtol,
                                     full_output=False, maxiter=10000)
                kl_div = kl(eta_opt)
                if np.abs(kl_div - max_kl) > 0.02 * max_kl:
                    diff = 5 * (eta_opt * rtol + xtol)
                    eta_opt = opt.bisect(lambda eta: kl(eta) - max_kl, eta_opt - diff, eta_opt + diff,
                                         full_output=False, maxiter=10000)
        else:
            eta_opt = 0
    except Exception as e:
        print("Warning! Exception during computation of new distribution, will use eta=" + str(max_eta) + "!")
        print(e)
        eta_opt = max_eta

    kl_div = kl(eta_opt)
    print("Eta: " + str(eta_opt), "KL Divergence: " + str(kl_div))

    return eta_opt


def central_differences(f, x, delta=1e-5):
    dim = x.shape[0]
    f_dim = f(x).shape
    grad = np.zeros(f_dim + (dim,))
    for i in range(0, dim):
        x[i] += delta
        f_upper = f(x)
        x[i] -= 2 * delta
        f_lower = f(x)
        x[i] += delta
        grad[..., i] = (f_upper - f_lower) / (2 * delta)
    return np.squeeze(grad)


def ridge_regression(phi, y, ridge_factor=1e-6):
    return np.linalg.solve(np.dot(phi.T, phi) + ridge_factor * np.eye(phi.shape[1]), np.dot(phi.T, y))


def sample_normal_with_bounds(mu, sigma, lower_bounds, upper_bounds, max_retries=100):
    if mu.shape[0] == 1:
        x = np.array([rand.normal(np.squeeze(mu), np.squeeze(np.sqrt(sigma)))])
    else:
        x = rand.multivariate_normal(mu, sigma)

    count = 0
    while np.any(x < lower_bounds) or np.any(x > upper_bounds):
        if count == max_retries:
            x = np.maximum(np.minimum(x, upper_bounds), lower_bounds)
            break

        if mu.shape[0] == 1:
            x = np.array([rand.normal(np.squeeze(mu), np.squeeze(np.sqrt(sigma)))])
        else:
            x = rand.multivariate_normal(mu, sigma)
        count += 1

    return x


def sample_with_bounds(sampling_func, lower_bounds, upper_bounds, max_retries=100):
    x = sampling_func()

    count = 0
    while np.any(x < lower_bounds) or np.any(x > upper_bounds):
        if count == max_retries:
            x = np.maximum(np.minimum(x, upper_bounds), lower_bounds)
            break

        x = sampling_func()
        count += 1

    return x


def load_pickle_file(directory, prefix, require_file=False, allow_none=False, with_filename=False):
    if not os.path.exists(directory):
        if require_file:
            raise RuntimeError("The given directory does not exist: " + str(directory))
        else:
            print("The given directory does not exist: " + str(directory))
            return None

    suffix = ".pkl"
    candidates = {}
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)) and file.startswith(prefix) and file.endswith(
                suffix):
            candidates[file[len(prefix):-len(suffix)]] = os.path.join(directory, file)

    if len(candidates) == 0:
        if require_file:
            raise RuntimeError("No files matching the criteria were contained in the given directory")
        else:
            print("No files matching the criteria were contained in the given directory")
            return None

    # If there are multiple matching files, we ask the user to specify the desired one
    if len(candidates) > 1:
        print("Multiple files found:")
        count = 0
        keys = []
        for k, v in candidates.items():
            print("[" + str(count) + "]: " + k.split('.')[0])
            keys.append(k)
            count += 1

        if allow_none:
            print("[" + str(count) + "]: None")

        choice = int(input("Please choose one of the files: "))

        if choice >= count or count < 0:
            if allow_none and choice == count:
                return None
            else:
                raise RuntimeError("Invalid index provided")

        candidate = candidates[keys[choice]]
    else:
        for k, v in candidates.items():
            candidate = v

        if allow_none:
            choice = int(input("One file found - do you want to load it [1/0]? "))
            if choice == 0:
                return None

    with open(candidate, "rb") as f:
        tpl = pickle.load(f)

    if with_filename:
        return tpl, candidate
    else:
        return tpl
