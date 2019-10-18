import argparse
import numpy as np
import sprl.environments as environments
from sprl.util.misc import load_pickle_file
import os
import datetime
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl
from functools import partial
from matplotlib2tikz import save as tikz_save
from scipy.signal import savgol_filter
from sprl.distributions.kl_joint import KLJoint

# We need to set this in order to be able to save the figures to PGF (at least on the laptop that we create the plots
# with)
rc_xelatex = {'pgf.rcfonts': False,
              'font.family': "serif"}
mpl.rcParams.update(rc_xelatex)


def compute_rewards_and_successes(env, policies):
    data = {}

    spec = env.get_spec()
    n_experiments = len(policies)

    # We sample the progress at 50 steps throughout training
    idx = [int(x) for x in np.linspace(0, len(policies[0]), 50, endpoint=False)]
    # We also ensure that the performance in the last iteration will be always computed
    if idx[-1] != len(policies[0]) - 1:
        idx.append(len(policies[0]) - 1)

    average_rewards = []
    average_success = []
    for i in range(0, n_experiments):
        print("Computing data for experiment " + str(i))
        cur_average_rewards = []
        cur_average_success = []
        for j in idx:
            __, __, rewards, success = env.sample_rewards(spec.n_samples, policies[i][j])
            cur_average_rewards.append(np.mean(rewards))
            cur_average_success.append(np.mean(success))
            print("\tIteration " + str(j) + ": " + str(cur_average_rewards[-1]))

        average_rewards.append(cur_average_rewards)
        average_success.append(cur_average_success)

    data["idx"] = idx
    data["rewards"] = np.array(average_rewards)
    data["successes"] = np.array(average_success)
    return data


def compute_visualization_data(env, log_dir, plot_log_dir, algorithm):
    # We first try load a plot log from disk
    print("Trying to load plot log for algorithm '" + algorithm + "' from disk")
    data = load_pickle_file(plot_log_dir, env.get_name() + "-" + algorithm + "-log-", allow_none=True)
    # If no such log exists, we need to compute the data ...
    if data is None:
        print("Not plot log found - computing data")
        tmp, file_name = load_pickle_file(log_dir, env.get_name() + "-" + algorithm + "-", require_file=True,
                                          with_filename=True)
        prefix = os.path.join(log_dir, env.get_name() + "-" + algorithm + "-")
        suffix = file_name[len(prefix):]

        if algorithm == "sprl":
            policies = []
            distributions = []
            for i in range(0, len(tmp)):
                sub_policies = []
                sub_distributions = []
                for j in range(0, len(tmp[i][0])):
                    if isinstance(tmp[i][0][j], KLJoint):
                        sub_policies.append(tmp[i][0][j].policy)
                        sub_distributions.append(tmp[i][0][j].distribution)
                    else:
                        sub_policies.append(tmp[i][0][j])
                policies.append(sub_policies)
                distributions.append(sub_distributions)
            # policies = [tpl[0].policy for tpl in tmp]
        else:
            policies = [tpl[0] for tpl in tmp]

        data = compute_rewards_and_successes(env, policies)
        if algorithm == "sprl":
            data["distributions"] = distributions

        if not os.path.exists(plot_log_dir):
            os.makedirs(plot_log_dir)

        log_file_name = env.get_name() + "-" + algorithm + "-log-" + suffix
        with open(os.path.join(plot_log_dir, log_file_name), "wb") as f:
            pickle.dump(data, f)

    return data


def visualize_perf_or_suc(idxs, perf_sucs, names, is_success, fontsize, clrs, ax):
    lines = []
    for i in range(0, len(idxs)):
        l0, = ax.plot(idxs[i], np.percentile(perf_sucs[i], 50, axis=0), linewidth=2, color=clrs[i])
        ax.fill_between(idxs[i], np.percentile(perf_sucs[i], 90, axis=0),
                        np.percentile(perf_sucs[i], 10, axis=0), alpha=0.5, color=l0.get_color())
        lines.append(l0)
    ax.legend(lines, names, prop={'size': fontsize})

    ylabel = "Success Rate" if is_success else "Reward"
    ax.set_ylabel(ylabel, rotation=90, fontsize=fontsize)
    ax.set_xlabel("Iterations", fontsize=fontsize)

    __ = [tick.label.set_fontsize(fontsize) for tick in ax.xaxis.get_major_ticks()]
    __ = [tick.label.set_fontsize(fontsize) for tick in ax.xaxis.get_minor_ticks()]
    __ = [tick.label.set_fontsize(fontsize) for tick in ax.yaxis.get_major_ticks()]
    __ = [tick.label.set_fontsize(fontsize) for tick in ax.yaxis.get_minor_ticks()]


def visualize_gaussian_entropy(distributions, fontsize, ax):
    entropies = []
    for dists in distributions:
        entropies_cur = []
        for dist in dists:
            sigma = dist.get_moments()[1]
            entropies_cur.append(0.5 * (np.log(np.linalg.det(sigma)) + sigma.shape[0] * (1 + np.log(2 * np.pi))))
        entropies.append(entropies_cur)

    l0, = ax.plot(np.linspace(0, len(entropies[0]) - 1, len(entropies[0])), np.percentile(entropies, 50, axis=0),
                  linewidth=2)
    ax.fill_between(np.linspace(0, len(entropies[0]) - 1, len(entropies[0])), np.percentile(entropies, 90, axis=0),
                    np.percentile(entropies, 10, axis=0), alpha=0.5, color=l0.get_color())

    ax.set_xlabel("Iteration", fontsize=fontsize)
    ax.set_ylabel("$H\\left(c\\right)$", fontsize=fontsize)
    __ = [tick.label.set_fontsize(fontsize) for tick in ax.xaxis.get_major_ticks()]
    __ = [tick.label.set_fontsize(fontsize) for tick in ax.xaxis.get_minor_ticks()]
    __ = [tick.label.set_fontsize(fontsize) for tick in ax.yaxis.get_major_ticks()]
    __ = [tick.label.set_fontsize(fontsize) for tick in ax.yaxis.get_minor_ticks()]


def visualize_distribution_evolution(env, idx, distributions, alphas, fontsize, xticks, yticks, xlabel, ylabel,
                                     bounds_overwrite, ax):
    jet = plt.get_cmap("hot")
    c_norm = colors.Normalize(vmin=0, vmax=1)
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=jet)

    if not isinstance(alphas, list):
        alphas = [alphas] * len(idx)

    for dist, alpha, c_val in zip([distributions[i] for i in idx], alphas, np.linspace(0, 0.6, len(idx))):
        line = dist.init_visualization(ax, alpha=alpha, fill=True, color=scalar_map.to_rgba(c_val))
        dist.update_visualization(line)

    spec = env.get_spec()
    line = spec.target_dist.init_visualization(ax, alpha=1, fill=False, color="black")
    spec.target_dist.update_visualization(line)
    line.set_linestyle("-.")

    if bounds_overwrite is None:
        lower_bounds = spec.target_dist.lower_bounds
        upper_bounds = spec.target_dist.upper_bounds
    else:
        lower_bounds = bounds_overwrite[0]
        upper_bounds = bounds_overwrite[1]
    ax.set_xlim([lower_bounds[0], upper_bounds[0]])

    if lower_bounds.shape[0] > 1:
        ax.set_ylim([lower_bounds[1], upper_bounds[1]])

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    __ = [tick.label.set_fontsize(fontsize) for tick in ax.xaxis.get_major_ticks()]
    __ = [tick.label.set_fontsize(fontsize) for tick in ax.xaxis.get_minor_ticks()]
    if xticks is not None:
        ax.xaxis.set_ticks(xticks)
        ax.xaxis.set_ticklabels(xticks)

    __ = [tick.label.set_fontsize(fontsize) for tick in ax.yaxis.get_major_ticks()]
    __ = [tick.label.set_fontsize(fontsize) for tick in ax.yaxis.get_minor_ticks()]
    if yticks is not None:
        ax.yaxis.set_ticks(yticks)
        ax.yaxis.set_ticklabels(yticks)

    scalar_map.set_array([])
    cbar = plt.colorbar(scalar_map, ax=ax, boundaries=np.arange(0, 0.7, .05), values=np.arange(0, 0.7, .05)[:-1],
                        alpha=1.8 * (alphas[0] if isinstance(alphas, list) else alphas))
    cbar.set_ticks(np.linspace(0, 0.6, len(idx)))
    cbar.set_ticklabels(idx)
    cbar.ax.tick_params(labelsize=fontsize)

    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(fontsize)
    ax.xaxis.get_offset_text().set_fontsize(fontsize)


def visualize_fn_wrapper(vis_fns):
    if len(vis_fns) == 1:
        f = plt.figure()
        vis_fns[0](f.gca())
    else:
        w, h = plt.figaspect(1)
        f, axes = plt.subplots(len(vis_fns), 1)
        f.set_figheight(h)
        f.set_figwidth(w)
        for i in range(0, len(vis_fns)):
            vis_fns[i](axes[i])


def visualize(envs, data_sprl, data_creps, data_cmaes, data_goalgan, data_saggriac, index_map, fontsize, save, plot_dir,
              suffix):
    save_prefix = ""
    for env in envs:
        save_prefix += env.get_name() + "+"
    save_prefix = save_prefix[0:-1]

    complete_data = (data_sprl, data_creps)
    labels = ["SPRL", "C-REPS"]
    colors = ["C0", "C1"]
    if data_goalgan is not None:
        complete_data += (data_goalgan,)
        labels.append("GoalGAN")
        colors.append("C2")
    if data_saggriac is not None:
        complete_data += (data_saggriac,)
        labels.append("SAGG-RIAC")
        colors.append("C3")
    if data_cmaes is not None:
        complete_data += (data_cmaes,)
        labels.append("CMA-ES")
        colors.append("C9")

    visualize_fn_wrapper([partial(visualize_perf_or_suc, [d_sub["idx"] for d_sub in d],
                                  [d_sub["rewards"] for d_sub in d], labels, False, fontsize, colors) for d in
                          zip(*complete_data)])
    plt.tight_layout()

    if save:
        if suffix == ".tex":
            tikz_save(os.path.join(plot_dir, save_prefix + "-rewards" + suffix))
        else:
            plt.savefig(os.path.join(plot_dir, save_prefix + "-rewards" + suffix))
    else:
        plt.show()

    visualize_fn_wrapper([partial(visualize_perf_or_suc, [d_sub["idx"] for d_sub in d],
                                  [d_sub["successes"] for d_sub in d], labels, False, fontsize, colors) for d in
                          zip(*complete_data)])
    plt.tight_layout()

    if save:
        if suffix == ".tex":
            tikz_save(os.path.join(plot_dir, save_prefix + "-successes" + suffix))
        else:
            plt.savefig(os.path.join(plot_dir, save_prefix + "-successes" + suffix))
    else:
        plt.show()

    if environment.get_spec().target_dist.lower_bounds.shape[0] <= 2:
        visualize_fn_wrapper(
            [partial(visualize_distribution_evolution, env, index_map[env.get_name()]["idx"],
                     ds["distributions"][index_map[env.get_name()]["experiment"]],
                     index_map[env.get_name()]["alphas"], fontsize,
                     index_map[env.get_name()]["x_ticks"], index_map[env.get_name()]["y_ticks"],
                     index_map[env.get_name()]["x_label"], index_map[env.get_name()]["y_label"],
                     index_map[env.get_name()]["bounds_overwrite"])
             for env, ds in zip(envs, data_sprl)])
        plt.tight_layout()
        if save:
            if suffix == ".tex":
                tikz_save(os.path.join(plot_dir, save_prefix + "-distribution-evolution" + suffix))
            else:
                plt.savefig(os.path.join(plot_dir, save_prefix + "-distribution-evolution" + suffix))
        else:
            plt.show()
    else:
        visualize_fn_wrapper(
            [partial(visualize_gaussian_entropy, ds["distributions"], fontsize) for ds in data_sprl])
        plt.tight_layout()
        if save:
            if suffix == ".tex":
                tikz_save(os.path.join(plot_dir, save_prefix + "-entropy" + suffix))
            else:
                plt.savefig(os.path.join(plot_dir, save_prefix + "-entropy" + suffix))
        else:
            plt.show()


def preprocess_cmaes_data(d_cmaes):
    offset = 0
    while d_cmaes[0]["idx"][offset] == 0.0:
        offset += 1

    itv = int(0.02 * len(d_cmaes[0]["idx"]))

    filtered_rewards = savgol_filter(np.array([dat["rewards"][offset:] for dat in d_cmaes]), 51, 3,
                                     axis=1)
    filtered_successes = savgol_filter(np.array([dat["successes"][offset:] for dat in d_cmaes]), 51, 3,
                                       axis=1)
    processed_d_cmaes = {"idx": np.concatenate((d_cmaes[0]["idx"][offset::itv], [d_cmaes[0]["idx"][-1]])),
                         "rewards": np.concatenate((filtered_rewards[:, 0::itv], filtered_rewards[:, -1][:, None]),
                                                   axis=1),
                         "successes": np.concatenate((filtered_successes[:, 0::itv], filtered_successes[:, -1][:, None]),
                                                     axis=1)}

    return processed_d_cmaes


def filter_best_n(best_n, data):
    if data is not None:
        for d in data:
            idx_sort = np.argsort(d["successes"][:, -1])
            d["successes"] = d["successes"][idx_sort[-best_n:], :]
            d["rewards"] = d["rewards"][idx_sort[-best_n:], :]

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize experiment results with with Self-Paced learning for C-REPS")
    parser.add_argument("--environment", type=str, choices=["gate", "reacher", "reacher-obstacle", "ball-in-a-cup"],
                        help='The environment for the experiment - the reacher experiments require a MuJoCo license')
    parser.add_argument("--n_cores", type=int, nargs="?", default=1,
                        help="The number of cores that will be used to compute the results")
    parser.add_argument("--setting", dest="settings", type=str, default=[], action="append",
                        help="The specific experiment setting - currently only 'precision' and 'global' are supported"
                             " for the 'gate' environments")
    parser.add_argument("--log_dir", nargs="?", default="logs", help="A path to a directory, in which the experiment"
                                                                     " data is stored")
    parser.add_argument("--plot_log_dir", nargs="?", default="plot_logs",
                        help="A path to a directory, in which the logging data is stored (plot logs will be of much "
                             "smaller size than the experiment logs).")
    parser.add_argument("--plot_dir", nargs="?", default="plots",
                        help="A path to a directory, in which the plots will be stored if specified")
    parser.add_argument("--suffix", nargs="?", default=".pgf",
                        help="The suffix of the stored plots - this suffix will decide over the picture format (e.g. "
                             "PDF, PNG, PFG, ...)")
    parser.add_argument("--fontsize", nargs="?", type=int, default=10, help="The fontsize of the plots")
    parser.add_argument("--store_plots", action="store_true", help="Store the plots instead of show them")
    parser.add_argument("--only_compute", action="store_true",
                        help="Only compute the plot data but do not visualize them")
    parser.add_argument("--add_cmaes", action="store_true", help="Also include the CMAES performance in the plots")
    parser.add_argument("--add_goalgan", action="store_true", help="Also include the CMAES performance in the plots")
    parser.add_argument("--add_saggriac", action="store_true", help="Also include the CMAES performance in the plots")
    parser.add_argument("--best_n", nargs="?", type=int, default=-1, help="Only plot the stastics of the n best trials")

    args = parser.parse_args()

    index_map = {
        "reacher-obstacle-default": {
            "experiment": 0, "idx": [10, 50, 110, 180, 300],
            "alphas": [0.3, 0.3, 0.3, 0.3, 0.5],
            "x_ticks": [0.02, 0.04, 0.06, 0.08],
            "x_label": "Size \#1",
            "y_label": "Size \#2",
            "y_ticks": [0.02, 0.04, 0.06, 0.08],
            "bounds_overwrite": None
        },
        "gate-precision": {
            "experiment": 0, "idx": [20, 80, 130, 210, 320],
            "alphas": 0.3,
            "x_ticks": None, "y_ticks": None,
            "x_label": "Gate Position",
            "y_label": "Gate Width",
            "bounds_overwrite": None
        },
        "gate-global": {
            "experiment": 0, "idx": [50, 150, 250, 400],
            "alphas": 0.3,
            "x_ticks": None, "y_ticks": None,
            "x_label": "Gate Position",
            "y_label": "Gate Width",
            "bounds_overwrite": None
        },
        "ball-in-a-cup-default": {
            "experiment": 0, "idx": [5, 30, 80, 120],
            "alphas": 0.3,
            "x_ticks": None, "y_ticks": None,
            "x_label": "Scale",
            "y_label": "PDF",
            "bounds_overwrite": None
        }
    }

    envs = []
    data_sprl = []
    data_creps = []
    data_cmaes = [] if args.add_cmaes else None
    data_goalgan = [] if args.add_goalgan else None
    data_saggriac = [] if args.add_saggriac else None
    if len(args.settings) == 0:
        environment = environments.get(args.environment, cores=args.n_cores)
        envs.append(environment)
        data_sprl.append(compute_visualization_data(environment, args.log_dir, args.plot_log_dir, "sprl"))
        data_creps.append(compute_visualization_data(environment, args.log_dir, args.plot_log_dir, "creps"))

        if args.add_cmaes:
            file_name = args.environment + "-" + environment.setting + "-cmaes-"
            data_cmaes.append(preprocess_cmaes_data(load_pickle_file(args.log_dir, file_name, require_file=True)))

        if args.add_goalgan:
            data_goalgan.append(compute_visualization_data(environment, args.log_dir, args.plot_log_dir, "goalgan"))

        if args.add_saggriac:
            data_saggriac.append(compute_visualization_data(environment, args.log_dir, args.plot_log_dir, "saggriac"))
    else:
        for setting in args.settings:
            environment = environments.get(args.environment, cores=args.n_cores)
            environment.set_setting(setting)

            envs.append(environment)
            data_sprl.append(compute_visualization_data(environment, args.log_dir, args.plot_log_dir, "sprl"))
            data_creps.append(compute_visualization_data(environment, args.log_dir, args.plot_log_dir, "creps"))

            if args.add_cmaes:
                file_name = args.environment + "-" + environment.setting + "-cmaes-"
                data_cmaes.append(preprocess_cmaes_data(load_pickle_file(args.log_dir, file_name, require_file=True)))

            if args.add_goalgan:
                data_goalgan.append(compute_visualization_data(environment, args.log_dir, args.plot_log_dir, "goalgan"))

            if args.add_saggriac:
                data_saggriac.append(
                    compute_visualization_data(environment, args.log_dir, args.plot_log_dir, "saggriac"))

    if not args.only_compute:
        if args.store_plots and not os.path.exists(args.plot_dir):
            os.makedirs(args.plot_dir)

        if args.best_n > 0:
            data_sprl = filter_best_n(args.best_n, data_sprl)
            data_cmaes = filter_best_n(args.best_n, data_cmaes)
            data_creps = filter_best_n(args.best_n, data_creps)
            data_goalgan = filter_best_n(args.best_n, data_goalgan)
            data_saggriac = filter_best_n(args.best_n, data_saggriac)

        visualize(envs, data_sprl, data_creps, data_cmaes, data_goalgan, data_saggriac, index_map, args.fontsize,
                  args.store_plots, args.plot_dir, args.suffix)
