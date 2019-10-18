import argparse
import sprl.util.gym_envs.reach_avoid_sb
import gym
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines.bench.monitor import Monitor
import os
import numpy as np


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


def get_exp_dir(main_dir):
    os.makedirs(main_dir, exist_ok=True)

    exp_number = 0
    for path in os.listdir(main_dir):
        if path.startswith("exp-"):
            num = int(path[4:])
            if num > exp_number:
                exp_number = num

    new_path = os.path.join(main_dir, "exp-" + str(exp_number + 1))
    os.makedirs(new_path, exist_ok=True)
    return new_path, exp_number + 1


def main(log_dir, easy, n_steps=450):
    exp_dir, seed_offset = get_exp_dir(
        os.path.join(log_dir, "reacher-obstacle-default" + ("-easy" if easy else "") + "-ppo"))
    print("Seed offset: " + str(seed_offset))

    log_path = os.path.join(exp_dir, "ppo-reach-avoid.log")
    avg_log_path = exp_dir
    if not os.path.exists(log_path):
        n_envs = 8
        env = VecNormalize(
            SubprocVecEnv([create_env_fn(seed_offset * n_envs + i, easy=easy) for i in range(0, n_envs)]),
            gamma=0.999)
        model = PPO2(policy='MlpPolicy', env=env, n_steps=n_steps, nminibatches=5, verbose=1, gamma=0.999,
                     noptepochs=15, ent_coef=1e-3, lam=1, policy_kwargs=dict(layers=[164, 164]))

        average_rewards = []

        def log_callback(local_vars, global_vars):
            avg_r = np.mean([ep_info['r'] for ep_info in local_vars["ep_info_buf"]])
            average_rewards.append(avg_r)
            return True

        # 3067500 = 409 iterations (400 + 9 for buffer initialization) * 50 trajectories * 150 timesteps
        model.learn(3067500, seed=seed_offset, callback=log_callback)
        model.save(log_path)
        env.save_running_average(avg_log_path)
        np.save(os.path.join(exp_dir, "rewards.npy"), np.array(average_rewards))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PPO on a step-based version of the reacher task")
    parser.add_argument("--log_dir", nargs="?", default="logs", help="A path to a directory, in which the experiment"
                                                                     " data is stored")
    parser.add_argument("--easy", action="store_true", help="Run in an easy version of the task")
    parser.add_argument("--n_experiments", type=int, nargs="?", default=10,
                        help="The number of experiments that should be run")
    args = parser.parse_args()

    for i in range(0, args.n_experiments):
        main(log_dir=args.log_dir, easy=args.easy)
