import argparse
import numpy as np
import copy
import datetime
import pickle
import time
import cma
import os
from mpi4py import MPI
from functools import partial
import sprl.environments as environments
from sprl.algorithms.creps import CREPS
from sprl.algorithms.sprl import SPRL
from sprl.util.visualization import Visualization
from sprl.util.experiencebuffer import ExperienceBuffer
from sprl.distributions.kl_joint import KLJoint
from sprl.algorithms.SaggRIAC import SaggRIAC
from sprl.algorithms.generator import StateGAN, StateCollection
import tensorflow as tf


def creps_iteration_function(env, spec, policies, average_rewards, average_successes, creps, policy, buffer,
                             seed, iteration):
    policies.append(copy.deepcopy(policy))

    contexts, thetas, rewards, successes = env.sample_rewards(spec.n_samples, policy)
    buffer.insert(contexts, thetas, rewards, successes)

    buffered_contexts, buffered_thetas, buffered_rewards, buffered_successes = buffer.get()
    average_rewards.append(np.mean(buffered_rewards))
    average_successes.append(np.mean(buffered_successes))
    weights = creps.reweight_samples(buffered_contexts, buffered_rewards, spec.eta)
    policy.refit(weights, buffered_contexts, buffered_thetas)

    print("Seed: %d, Iteration: %3d, Reward: %4.2f, Success Rate: %1.2f" % (
        seed, iteration + 1, average_rewards[-1], average_successes[-1]))

    return weights, buffered_contexts, buffered_thetas, buffered_rewards


def sprl_iteration_function(env, spec, policies, average_rewards, average_successes, alg, distribution, buffer, seed,
                            iteration):
    policies.append(copy.deepcopy(distribution))

    contexts, thetas, rewards, successes = env.sample_rewards(spec.n_samples, distribution.policy,
                                                              distribution.distribution)
    buffer.insert(contexts, thetas, rewards, successes)

    buffered_contexts, buffered_thetas, buffered_rewards, buffered_successes = buffer.get()
    average_rewards.append(np.mean(buffered_rewards))
    average_successes.append(np.mean(buffered_successes))

    avg_rew = np.mean(buffered_rewards)
    kl_div = np.mean(distribution.distribution.get_log_pdf(buffered_contexts) -
                     spec.target_dist.get_log_pdf(buffered_contexts))
    alpha = spec.alpha_schedule.get_value(iteration, avg_rew, kl_div)
    weights, context_weights = alg.reweight_samples(buffered_contexts, distribution.distribution.get_log_pdf,
                                                    buffered_rewards, spec.eta, alpha)
    distribution.refit(weights, buffered_contexts, buffered_thetas, x_weights=context_weights)
    if spec.kl_div_thresh is not None and kl_div > spec.kl_div_thresh:
        var = np.diag(distribution.distribution.sigma)
        clipped_var = np.maximum(var, spec.lower_variance_bound)
        if np.any(clipped_var - var > 0.):
            print("Clipping Sigma to a minimum value")
        distribution.distribution.sigma[np.diag_indices_from(distribution.distribution.sigma)] = clipped_var

    print("Seed: %d, Iteration: %3d, KL-Div: %.3E, Reward: %4.2f, Success Rate: %1.2f" % (
        seed, iteration + 1, kl_div, average_rewards[-1], average_successes[-1]))

    return weights, buffered_contexts, buffered_thetas, buffered_rewards


def sagg_riac_iteration_function(env, spec, policies, average_rewards, average_successes, alg, sr, policy, buffer, seed,
                                 iteration):
    contexts = np.array(sr.sample_states(spec.n_samples))
    thetas = np.array([policy.sample_action(contexts[k, :]) for k in range(0, spec.n_samples)])
    rewards, successes = env.evaluate(contexts, thetas)

    sr.add_states(contexts, rewards)
    buffer.insert(contexts, thetas, rewards, successes)

    buffered_contexts, buffered_thetas, buffered_rewards, buffered_successes = buffer.get()

    policies.append(copy.deepcopy(policy))
    average_rewards.append(np.mean(buffered_rewards))
    average_successes.append(np.mean(buffered_successes))

    print("Seed: %d, Iteration: %3d, Reward: %4.2f, Success Rate: %1.2f" % (
        seed, iteration + 1, average_rewards[-1], average_successes[-1]))

    alg._feature_mean = np.mean(spec.value_features(buffered_contexts), axis=0)
    weights = alg.reweight_samples(buffered_contexts, buffered_rewards, spec.eta)
    policy.refit(weights, buffered_contexts, buffered_thetas)

    return weights, buffered_contexts, buffered_thetas, buffered_rewards


def goal_gan_iteration_function(env, spec, policies, average_rewards, average_successes, alg, gan, policy, buffer,
                                gan_buffer, success_buffer, lb, ub, n_old_samples, n_context_samples, n_resamples, seed,
                                iteration):
    contexts, thetas, rewards, successes, success_rates, labels, old_contexts, old_thetas, old_rewards, \
    old_successes = gan_policy_rollout(gan, policy, env, spec, success_buffer, lb, ub, n_old_samples,
                                       n_context_samples, n_resamples)
    buffer.insert(np.concatenate((contexts, old_contexts), axis=0),
                  np.concatenate((thetas, old_thetas), axis=0),
                  np.concatenate((rewards, old_rewards)), np.concatenate((successes, old_successes)))
    success_buffer.append(np.copy(contexts[0:n_context_samples])[success_rates == 1., :])
    gan_buffer.insert(np.copy(contexts[0:n_context_samples]), labels[:, None])

    buffered_contexts, buffered_thetas, buffered_rewards, buffered_successes = buffer.get()

    policies.append(copy.deepcopy(policy))
    average_rewards.append(np.mean(buffered_rewards))
    average_successes.append(np.mean(buffered_successes))

    print("Seed: %d, Iteration: %3d, Reward: %4.2f, Success Rate: %1.2f" % (
        seed, iteration + 1, average_rewards[-1], average_successes[-1]))

    alg._feature_mean = np.mean(spec.value_features(buffered_contexts), axis=0)
    weights = alg.reweight_samples(buffered_contexts, buffered_rewards, spec.eta)
    policy.refit(weights, buffered_contexts, buffered_thetas)

    gan_contexts, gan_labels = gan_buffer.get()
    if iteration % spec.goal_gan_spec.gan_train_interval == 0:
        if np.mean(gan_labels) == 0.:
            print("Skipping GAN training, as no successful examples are present")
        else:
            gan.train(gan_contexts, gan_labels, spec.goal_gan_spec.gan_train_iters)
    else:
        print("Skipping GAN training")

    return weights, buffered_contexts, buffered_thetas, buffered_rewards


def gan_policy_rollout(gan, policy, env, spec, success_buffer, lb, ub, n_old_samples, n_context_samples, n_resamples):
    # Run the experiments
    contexts = gan.sample_states_with_noise(n_context_samples)[0]
    resample_ids = np.random.permutation(n_context_samples)[0:n_resamples]
    contexts = np.concatenate((contexts, contexts, contexts[resample_ids], contexts[resample_ids]), axis=0)
    thetas = np.array([policy.sample_action(contexts[k, :]) for k in range(0, contexts.shape[0])])
    rewards, successes = env.evaluate(contexts, thetas)

    old_contexts = success_buffer.sample(n_old_samples, replay_noise=spec.goal_gan_spec.state_noise_level * (ub - lb))
    rem = n_old_samples - old_contexts.shape[0]
    if rem == n_old_samples:
        old_contexts = np.random.uniform(lb, ub, size=(rem, len(lb)))
    else:
        old_contexts = np.concatenate((old_contexts, np.random.uniform(lb, ub, size=(rem, len(lb)))), axis=0)
    old_thetas = np.array([policy.sample_action(old_contexts[k, :]) for k in range(0, n_old_samples)])
    old_rewards, old_successes = env.evaluate(old_contexts, old_thetas)

    # Compute the success rate of the individual contexts
    try:
        success_rates_tmp = successes[0:n_context_samples] + successes[n_context_samples:2 * n_context_samples]
        for i in range(0, n_resamples):
            success_rates_tmp[resample_ids[i]] += successes[2 * n_context_samples + i] + \
                                                  successes[2 * n_context_samples + n_resamples + i]
        # success_rates[resample_ids] += successes[2 * n_context_samples: 2 * n_context_samples + n_resamples]
        # success_rates[resample_ids] += successes[2 * n_context_samples + n_resamples:]
        divisors = 2. * np.ones(n_context_samples)
        divisors[resample_ids] = 4.

        success_rates = success_rates_tmp / divisors
    except Exception as e:
        print(successes)
        print(success_rates)
        print(divisors)

        raise e

    # Compute the labels and store data
    labels = np.logical_and(0.1 < success_rates, success_rates < 0.9).astype(np.float)
    return contexts, thetas, rewards, successes, success_rates, labels, old_contexts, old_thetas, old_rewards, \
           old_successes


def run_experiment(env, seed, visualize, algorithm):
    print("Running experiment " + str(seed + 1))
    np.random.seed(seed)
    env.set_seed(seed)

    spec = env.get_spec()

    # CMA-ES has a bit of a special-treatment
    if algorithm == "cmaes":
        opts = cma.CMAOptions()

        opts["bounds"] = [spec.init_policy.lower_bounds, spec.init_policy.upper_bounds]
        opts["maxfevals"] = (spec.n_iter + spec.buffer_size) * spec.n_samples
        opts["verbose"] = 1
        opts["tolstagnation"] = int(1e6)
        # We need the +1 here since CMA-ES rejects zero seeds
        opts["seed"] = seed + 1
        # We are only allowed to specify one variance for all variables, so we need to take the maximum to not
        # shrink the search space
        alg = cma.CMAEvolutionStrategy(spec.init_policy._mu, np.sqrt(np.max(np.diag(spec.init_policy._sigma))), opts)

        idx = []
        rewards = []
        successes = []
        theta_history = []
        count = 0
        while "maxfevals" not in alg.stop():
            thetas = alg.ask()
            contexts = np.array([spec.target_dist.get_moments()[0] for i in range(len(thetas))])
            r, s = env.evaluate(contexts, np.array(thetas))

            idx.append(np.maximum(0., (float(count) - float(spec.n_samples * (spec.buffer_size - 1))) / float(
                spec.n_samples)))
            count += len(thetas)

            rewards.append(np.mean(r))
            successes.append(np.mean(s))
            theta_history.append(np.array(thetas))

            print("Seed: %d, Count: %d, Reward: %4.2f, Success Rate: %1.2f" % (seed, count, np.mean(r), np.mean(s)))

            alg.tell(thetas, list(-r))
            alg.disp()

        log_data = {"idx": idx, "rewards": rewards, "successes": successes, "thetas": theta_history}
    # The remaining algorithms can be treated quite uniformly except for their setup
    else:
        t_start = time.time()
        policies = []
        average_rewards = []
        average_successes = []

        if algorithm == "goalgan":
            if isinstance(spec.init_dist, KLJoint):
                lb = np.copy(spec.init_dist.distribution.lower_bounds)
                ub = np.copy(spec.init_dist.distribution.upper_bounds)
            else:
                lb = np.copy(spec.init_dist.lower_bounds)
                ub = np.copy(spec.init_dist.upper_bounds)

            n_old_samples = int(spec.goal_gan_spec.p_old_samples * spec.n_samples)
            n_samples = spec.n_samples - n_old_samples

            n_context_samples = int(spec.goal_gan_spec.p_context_samples * n_samples)
            # We allow GoalGAN for one more sample in case the samples are not evenly dividable
            n_resamples = int(np.ceil((n_samples - 2 * n_context_samples) / 2))

            tf_session = tf.Session()
            gan = StateGAN(
                state_size=len(lb),
                evaluater_size=1,
                state_range=0.5 * (ub - lb),
                state_center=lb + 0.5 * (ub - lb),
                state_noise_level=(spec.goal_gan_spec.state_noise_level * (ub - lb))[None, :],
                generator_layers=[256, 256],
                discriminator_layers=[128, 128],
                noise_size=lb.shape[0],
                tf_session=tf_session,
                configs={"supress_all_logging": True}
            )

            tf_session.run(tf.initialize_local_variables())
            gan.pretrain_uniform(outer_iters=spec.goal_gan_spec.gan_pre_train_iters)

            alg = CREPS(spec.value_features, None, spec.regularizer)
            policy = copy.deepcopy(spec.init_policy)
            distribution = None
            buffer = ExperienceBuffer(spec.goal_gan_spec.buffer_size, 4)
            gan_buffer = ExperienceBuffer(spec.goal_gan_spec.buffer_size, 2)
            success_buffer = StateCollection(1,
                                             spec.goal_gan_spec.state_distance_threshold * np.linalg.norm(ub - lb))

            # Fill the initial buffer
            for j in range(0, spec.buffer_size - 1):
                contexts, thetas, rewards, successes, success_rates, labels, old_contexts, old_thetas, old_rewards, \
                old_successes = gan_policy_rollout(gan, policy, env, spec, success_buffer, lb, ub, n_old_samples,
                                                   n_context_samples, n_resamples)

                buffer.insert(np.concatenate((contexts, old_contexts), axis=0),
                              np.concatenate((thetas, old_thetas), axis=0),
                              np.concatenate((rewards, old_rewards)), np.concatenate((successes, old_successes)))
                success_buffer.append(contexts[0:n_context_samples][success_rates == 1., :])
                gan_buffer.insert(contexts[0:n_context_samples], labels[:, None])

            it_fn = partial(goal_gan_iteration_function, env, spec, policies, average_rewards, average_successes, alg,
                            gan, policy, buffer, gan_buffer, success_buffer, lb, ub, n_old_samples, n_context_samples,
                            n_resamples, seed)
        elif algorithm == "saggriac":
            if isinstance(spec.init_dist, KLJoint):
                lb = np.copy(spec.init_dist.distribution.lower_bounds)
                ub = np.copy(spec.init_dist.distribution.upper_bounds)
            else:
                lb = np.copy(spec.init_dist.lower_bounds)
                ub = np.copy(spec.init_dist.upper_bounds)

            sr = SaggRIAC(len(lb), state_bounds=np.stack((lb, ub)), state_center=lb + ((ub - lb) / 2.),
                          max_goals=spec.sagg_riac_spec.max_goals, max_history=spec.sagg_riac_spec.max_history)
            policy = copy.deepcopy(spec.init_policy)
            distribution = None
            alg = CREPS(spec.value_features, None, spec.regularizer)
            buffer = ExperienceBuffer(spec.buffer_size, 4)

            # Create the initial experience
            for j in range(0, spec.buffer_size - 1):
                contexts = np.array(sr.sample_states(spec.n_samples))

                thetas = np.array([policy.sample_action(contexts[k, :]) for k in range(0, spec.n_samples)])
                rewards, successes = env.evaluate(contexts, thetas)

                sr.add_states(contexts, rewards)
                buffer.insert(contexts, thetas, rewards, successes)

            it_fn = partial(sagg_riac_iteration_function, env, spec, policies, average_rewards, average_successes, alg,
                            sr, policy, buffer, seed)
        else:
            if algorithm == "creps":
                feature_mean = np.mean(spec.value_features(spec.target_dist.sample(n_samples=10 * spec.n_samples)),
                                       axis=0)
                alg = CREPS(spec.value_features, feature_mean, spec.regularizer)
            else:
                alg = SPRL(spec.value_features, spec.target_dist.get_log_pdf, spec.regularizer)

            # We copy the initial distribution and the policy since we may run multiple iterations
            if algorithm == "creps":
                distribution = None
                policy = copy.deepcopy(spec.init_policy)
            else:
                itl_distribution = copy.deepcopy(spec.init_dist)
                distribution = itl_distribution.distribution
                policy = itl_distribution.policy

            # We initialize the buffer with data
            buffer = ExperienceBuffer(spec.buffer_size, 4)
            for j in range(0, spec.buffer_size - 1):
                buffer.insert(*env.sample_rewards(spec.n_samples, policy, distribution))

            if algorithm == "creps":
                it_fn = partial(creps_iteration_function, env, spec, policies, average_rewards, average_successes, alg,
                                policy, buffer, seed)
            else:
                it_fn = partial(sprl_iteration_function, env, spec, policies, average_rewards, average_successes, alg,
                                itl_distribution, buffer, seed)

        # This is the actual main loop
        if visualize:
            vis = Visualization(spec.target_dist, policy, distribution)
            vis.visualize(spec.n_iter, it_fn)
        else:
            for j in range(0, spec.n_iter):
                it_fn(j)

        policies.append(copy.deepcopy(policy))

        t_end = time.time()

        __, __, rewards, successes = env.sample_rewards(spec.n_samples, policy)
        print("Seed: %d, Final Reward: %4.2f, Final Success Rate: %1.2f, Training Time: %.2E" % (
            seed, np.mean(rewards), np.mean(successes), t_end - t_start))

        log_data = (policies, average_rewards, average_successes)

    # If we used GoalGAN, we need to close the session and reset the graph for the next run
    if algorithm == "goalgan":
        tf_session.close()
        tf.reset_default_graph()

    return log_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with with Self-Paced learning for C-REPS")
    parser.add_argument("--environment", type=str, nargs="?", default="gate",
                        choices=["gate", "reacher-obstacle", "ball-in-a-cup", "point-mass"],
                        help='The environment for the experiment - the reacher experiments require a MuJoCo license')
    parser.add_argument("--setting", type=str, default="",
                        help="The specific experiment setting - currently only 'precision' and 'global' are supported"
                             " for the 'gate' environment")
    parser.add_argument("--n_experiments", type=int, nargs="?", default=40, help="The number of experiments to be run")
    parser.add_argument("--n_cores", type=int, nargs="?", default=1,
                        help="The number of cores per MPI process that will be used to compute the experiments")
    parser.add_argument("--n_mpi", type=int, nargs="?", default=1,
                        help="The number of independent MPI processes that will be used to run different experiments")
    parser.add_argument("--log_dir", nargs="?", default="logs", help="A path to a directory, in which the generated"
                                                                     " data will be stored for later visualization")
    parser.add_argument("--log_rewards", action="store_true",
                        help="Write the acquired rewards and successes into the log")
    parser.add_argument("--log_buffer", action="store_true",
                        help="Write the data available for learning in every iteration - needs much more storage")
    parser.add_argument("--visualize", action="store_true", help="Visualize the learning progress")
    parser.add_argument("--algorithm", type=str, nargs="?", default="sprl", choices=["sprl", "cmaes", "creps",
                                                                                     "saggriac", "goalgan"],
                        help="The algorithm with which to run the experiment")

    args = parser.parse_args()

    environment = environments.get(args.environment, cores=args.n_cores)
    if args.setting != "":
        environment.set_setting(args.setting)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("Running with " + str(size) + " MPI workers!")

    # Compute the number of experiments to run
    n_local_exp = int(args.n_experiments / size) + (1 if rank < args.n_experiments % size else 0)
    seed_offset = rank * int(args.n_experiments / size) + min(rank, args.n_experiments % size)
    logs = []
    for i in range(0, n_local_exp):
        logs.append(run_experiment(environment, seed_offset + i, args.visualize, args.algorithm))

    # After the experiments, rank 0 collects all the logs into one big list
    if rank == 0:
        for i in range(1, size):
            logs += comm.recv(source=i)

        if len(logs) != args.n_experiments:
            raise RuntimeError("Something went wrong with distributing the experiments among MPI Workers!")

        # Finally, we store the result
        log_file_name = environment.get_name() + "-" + args.algorithm + "-" + datetime.datetime.now().isoformat() + ".pkl"
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        with open(os.path.join(args.log_dir, log_file_name), "wb") as f:
            pickle.dump(logs, f)
    else:
        comm.send(logs, dest=0)
