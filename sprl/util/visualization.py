import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec


class Visualization:

    def __init__(self, target_dist, policy, dist=None):
        self.target_dist = target_dist
        self.sampling_dist = dist
        self.policy = policy
        self.lower_bounds = self.target_dist.lower_bounds
        self.upper_bounds = self.target_dist.upper_bounds

    @staticmethod
    def compute_policy_prediction_error(policy, contexts, thetas, weights):
        n_samples = weights.shape[0]
        error = 0
        for i in range(0, n_samples):
            error += weights[i] * np.sum(np.square(thetas[i] - policy.compute_greedy_action(contexts[i])))
        error /= n_samples
        return np.sqrt(error)

    @staticmethod
    def compute_policy_variance(policy, contexts):
        variances = []
        for i in range(0, contexts.shape[0]):
            variances.append(np.mean(np.diag(policy.compute_variance(contexts[i]))))
        return np.mean(variances)

    @staticmethod
    def update_plot(lines, ax, average_rewards, prediction_errors, policy_variances, weights,
                    contexts, rewards, target_dist, sampling_dist):
        line_episode_reward, line_average_reward, line_policy_error, line_pol_var, line_samples, \
        line_target_dist, line_sampling_dist = lines
        ax_episode_reward, ax_average_reward, ax_policy_error, ax_pol_var, ax_distribution = ax

        idx = np.argsort(weights)
        line_episode_reward.set_data(weights[idx], rewards[idx])
        ax_episode_reward.set_xlim(np.min(weights), np.max(weights))
        ax_episode_reward.set_ylim(np.min(rewards), np.max(rewards))

        line_average_reward.set_data(np.arange(1, len(average_rewards) + 1), average_rewards)
        ax_average_reward.set_ylim(np.min(average_rewards), np.max(average_rewards))

        line_policy_error.set_data(np.arange(1, len(prediction_errors) + 1), prediction_errors)
        ax_policy_error.set_ylim(0, np.max(prediction_errors))

        if contexts.shape[1] == 1:
            line_samples.set_offsets(np.concatenate((contexts, np.zeros_like(contexts)), axis=1))
        else:
            line_samples.set_offsets(contexts)
        line_samples.set_array(rewards)
        line_samples.set_clim(vmin=np.min(rewards), vmax=np.max(rewards))
        target_dist.update_visualization(line_target_dist)
        if sampling_dist is not None:
            sampling_dist.update_visualization(line_sampling_dist)

        line_pol_var.set_data(np.arange(1, len(prediction_errors) + 1), policy_variances)
        ax_pol_var.set_ylim(np.min(policy_variances), np.max(policy_variances))

    @staticmethod
    def init_plotting(lines):
        line_episode_reward, line_average_reward, line_policy_error, line_pol_var, line_samples, \
        line_target_dist, line_sampling_dist = lines

        line_episode_reward.set_data([], [])
        line_average_reward.set_data([], [])
        line_policy_error.set_data([], [])
        line_samples.set_offsets(np.zeros((0, 2)))
        line_pol_var.set_data([], [])

        if line_sampling_dist is None:
            return line_episode_reward, line_average_reward, line_policy_error, line_pol_var, line_samples, \
                   line_target_dist
        else:
            return lines

    def wrap_iteration(self, iteration, iteration_func, lines, ax, average_rewards, prediction_errors,
                       policy_variances):
        iteration_res = iteration_func(iteration)
        weights, contexts, thetas, rewards = iteration_res

        average_rewards.append(np.mean(rewards))
        prediction_errors.append(Visualization.compute_policy_prediction_error(self.policy, contexts, thetas, weights))
        policy_variances.append(Visualization.compute_policy_variance(self.policy, contexts))

        self.update_plot(lines, ax, average_rewards, prediction_errors, policy_variances, weights,
                         contexts, rewards, self.target_dist, self.sampling_dist)

        return lines

    def visualize(self, n_iterations, iteration_function):
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        gs = GridSpec(4, 3)

        ax_episode_reward = fig.add_subplot(gs[0, :])
        ax_episode_reward.set_ylabel("Episode Reward")
        ax_episode_reward.set_xlabel("Weight")
        line_episode_reward, = ax_episode_reward.plot([], [])
        ax_episode_reward.set_xlim(0, 2)
        ax_episode_reward.set_ylim(-100, 0)

        ax_pol_var = fig.add_subplot(gs[1, 0])
        ax_pol_var.set_ylabel("Log Policy Variance")
        ax_pol_var.set_xlabel("Iterations")
        line_pol_var, = ax_pol_var.semilogy([], [])
        ax_pol_var.set_xlim(1, n_iterations)
        ax_pol_var.set_ylim(1e-2, 2)

        ax_average_reward = fig.add_subplot(gs[3, 0])
        ax_average_reward.set_xlabel("Iterations")
        ax_average_reward.set_ylabel("Average Reward")
        line_average_reward, = ax_average_reward.plot([], [])
        ax_average_reward.set_xlim(0, n_iterations)
        ax_average_reward.set_ylim(-100, 0)

        ax_policy_error = fig.add_subplot(gs[2, 0])
        ax_policy_error.set_ylabel("Policy Error")
        ax_policy_error.set_xlabel("Iterations")
        line_policy_error, = ax_policy_error.plot([], [])
        ax_policy_error.set_xlim(1, n_iterations)
        ax_policy_error.set_ylim(0, 2)

        ax_distribution = fig.add_subplot(gs[1:, 1:])
        ax_distribution.set_title("Rewards")
        cm = plt.cm.get_cmap('RdYlBu')
        line_samples = ax_distribution.scatter([], [], c=[], vmin=-5, vmax=0, s=35, cmap=cm)
        if self.lower_bounds.shape[0] == 1:
            ax_distribution.set_xlim(self.lower_bounds[0], self.upper_bounds[0])
            ax_distribution.set_ylim(-0.25, 1.25)
        else:
            ax_distribution.set_xlim(self.lower_bounds[0], self.upper_bounds[0])
            ax_distribution.set_ylim(self.lower_bounds[1], self.upper_bounds[1])
        plt.colorbar(line_samples)

        line_target_dist = self.target_dist.init_visualization(ax_distribution, "C0")
        if self.sampling_dist is not None:
            line_sampling_dist = self.sampling_dist.init_visualization(ax_distribution, "C1")
            ax_distribution.legend((line_target_dist, line_sampling_dist),
                                   ('Goal Distribution', 'Sampling Distribution'))
        else:
            line_sampling_dist = None
            ax_distribution.legend((line_target_dist,), ('Goal Distribution',))

        lines = line_episode_reward, line_average_reward, line_policy_error, line_pol_var, line_samples, \
                line_target_dist, line_sampling_dist
        ax = ax_episode_reward, ax_average_reward, ax_policy_error, ax_pol_var, ax_distribution

        plt.tight_layout()

        average_rewards = []
        prediction_errors = []
        policy_variances = []

        ani = animation.FuncAnimation(fig, lambda i: self.wrap_iteration(i, iteration_function, lines, ax,
                                                                         average_rewards, prediction_errors,
                                                                         policy_variances),
                                      n_iterations, lambda: self.init_plotting(lines), blit=False, interval=10,
                                      repeat=False)
        plt.show()
