import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sprl.util.gym_envs.reach_avoid
import sprl.util.gym_envs.reach_avoid_sb
from matplotlib2tikz.save import save as tikz_save
from sprl.util.misc import load_pickle_file
import gym
from sprl.util.det_promp import DeterministicProMP
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.bench.monitor import Monitor
import argparse
import os
from PIL import Image

# We need to set this in order to be able to save the figures to PGF (at least on the laptop that we create the plots
# with)
rc_xelatex = {'pgf.rcfonts': False,
              'font.family': "serif"}
mpl.rcParams.update(rc_xelatex)


def create_env_fn(seed, monitored=True, easy=True):
    def f():
        if easy:
            env = gym.make("FetchReachAvoidSBEasy-v1")
        else:
            env = gym.make("FetchReachAvoidSB-v1")
        env.seed(seed)
        if monitored:
            return Monitor(env, None)
        else:
            return env

    return f


def run_ppo_policy(env, exp_dir):
    log_path = os.path.join(exp_dir, "ppo-reach-avoid.log")
    env.load_running_average(exp_dir)
    model = PPO2.load(log_path, env=env)

    obs = env.reset()
    done = False

    states_cur = [env.get_original_obs()[0, [0, 2]]]
    while not done:
        obs, reward, done, info = env.step(model.predict(obs, deterministic=False)[0])
        if info[0]["is_collision"]:
            return states_cur
        else:
            states_cur.append(env.get_original_obs()[0, [0, 2]])
    env.close()

    return states_cur


def run_ppo_policies(easy, main_dir, n_exps):
    env = VecNormalize(DummyVecEnv([create_env_fn(0, monitored=False, easy=easy)]), gamma=0.999,
                       training=False)

    states = []
    for i in range(1, n_exps + 1):
        states.append(np.array(run_ppo_policy(env, os.path.join(main_dir, "exp-" + str(i)))))

    return states


def run_promp_policy(env, context, theta):
    obs = env.reset()
    n_steps = env._max_episode_steps
    actual_env = env.env
    actual_env._set_obstacle_information(context)

    weights = np.reshape(theta, (-1, 2))
    pmp = DeterministicProMP(n_basis=weights.shape[0])
    pmp.set_weights(float(n_steps), weights)
    actions = pmp.compute_trajectory(1, 1)[1]

    done = False
    step = 0
    states_cur = [obs["achieved_goal"][0:2]]
    while not done:
        obs, reward, done, info = env.step(actions[step, :])
        states_cur.append(obs["achieved_goal"][0:2])
        step += 1

    return states_cur


def run_promp_policies(context, thetas):
    env = gym.make("FetchReachAvoid-v1")

    states = []
    for i in range(0, len(thetas)):
        states.append(np.array(run_promp_policy(env, context, thetas[i])))

    return states


def visualize_policy(trajectories, names, colors, store_path=None):
    background_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "reacher-obstacle-background.png")
    plt.figure()

    lines = []
    for i in range(0, len(trajectories)):
        trajs = trajectories[i]
        for j in range(0, len(trajs)):
            l0, = plt.plot(trajs[j][:, 1], trajs[j][:, 0], linewidth=10, alpha=0.5, c=colors[i])
            if j == 0:
                lines.append(l0)

    plt.legend(lines, names)

    plt.xticks([], [])
    plt.yticks([], [])

    im = np.array(Image.open(background_file))
    plt.imshow(im, zorder=0, extent=[0.35, 1.15, 1.5, 1.1])

    plt.tight_layout()

    if store_path is None:
        plt.show()
    else:
        tikz_save(store_path)


def main():
    parser = argparse.ArgumentParser(description="Visualize final policies for the reacher-obstacle task")
    parser.add_argument("--log_dir", nargs="?", default="logs", help="A path to a directory, in which the experiment"
                                                                     " data is stored")
    parser.add_argument("--plot_dir", nargs="?", default="plots",
                        help="A path to a directory, in which the plots will be stored if specified")
    parser.add_argument("--store_plots", action="store_true", help="Store the plots as tikz plots instead of showing")

    args = parser.parse_args()

    context = np.array([0.08, 0.08])
    algs = ["sprl", "creps", "goalgan", "saggriac"]

    states = []
    for alg in algs:
        log = load_pickle_file(args.log_dir, "reacher-obstacle-default-" + alg)

        thetas = []
        for i in range(0, 10):
            thetas.append(log[i][0][-1].sample_action(context))
        states.append(run_promp_policies(context, thetas))

    store_path = os.path.join(args.plot_dir, "reacher-obstacle-policies-1.tex") if args.store_plots else None
    visualize_policy(states[0:-1], ["SPRL", "C-REPS", "GoalGAN"], ["C0", "C1", "C2"], store_path=store_path)

    thetas_cmaes = [l["thetas"][-1][0, :] for l in load_pickle_file(args.log_dir, "reacher-obstacle-default-cmaes")]
    states_cmaes = run_promp_policies(context, thetas_cmaes)
    states_ppo = run_ppo_policies(False, os.path.join(args.log_dir, "reacher-obstacle-default-ppo"), 10)

    store_path = os.path.join(args.plot_dir, "reacher-obstacle-policies-2.tex") if args.store_plots else None
    visualize_policy([states_cmaes, states_ppo, states[-1]], ["CMA-ES", "PPO", "SAGG-RIAC"], ["C9", "C8", "C3"],
                     store_path=store_path)


if __name__ == "__main__":
    main()
