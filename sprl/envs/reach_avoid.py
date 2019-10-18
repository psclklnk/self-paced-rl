from sprl.envs.abstractenv import AbstractEnvironment
from sprl.envs.spec import ExperimentSpec, GoalGANSpec, SAGGRiacSpec
from sprl.util.schedule import PercentageSchedule
from sprl.distributions.gaussian import Gaussian
from sprl.features.rbf import RadialBasisFunctions
from sprl.features.polynomial import PolynomialFeatures
from sprl.util.det_promp import DeterministicProMP
from sprl.distributions.kl_joint import KLJoint
from sprl.policies.kl_wlr import KLWLRPolicy
import copy
import sprl.util.gym_envs.reach_avoid
import numpy as np
import gym


class ReachAvoidCostFunction:

    def __init__(self, pid):
        self.pid = pid
        self.env = gym.make("FetchReachAvoid-v1")
        self.env.env.reward_type = "dense"

    def __call__(self, cs, xs):
        rewards, success, action_costs = self._run_experiment(cs, xs)
        return 10 * (2 * rewards[-1] - 1e-2 * action_costs), success

    def set_seed(self, seed):
        print("Process " + str(self.pid) + " - Setting seed " + str(seed))
        np.random.seed(seed)
        self.env.seed(seed)

    def _run_experiment(self, context, theta):
        pmp = DeterministicProMP(n_basis=20)
        pmp.set_weights(float(self.env._max_episode_steps), np.reshape(theta, (20, -1)))
        actions = pmp.compute_trajectory(1, 1)[1]

        done = False
        self.env.reset()
        self.env.env._set_obstacle_information(context)
        rewards = []
        k = 0
        while not done:
            obs, reward, done, info = self.env.step(actions[k, :])

            # This is just a sanity check to ensure that the context is really used
            if np.max(obs["obstacle_information"] - context) > 0:
                raise RuntimeError("Desired Goal is not equal to the sampled one")

            k += 1
            rewards.append(reward)

        return rewards, info["is_success"], np.sum(np.sum(np.square(actions), axis=1), axis=0)


class ReachAvoid(AbstractEnvironment):

    def __init__(self, name, n_cores):
        theta_dim = 40
        theta_lower_bounds = np.ones(theta_dim) * -100
        theta_upper_bounds = np.ones(theta_dim) * 100

        context_lower_bounds = np.array([0.01, 0.01])
        context_upper_bounds = np.array([0.09, 0.09])
        context_width = (context_upper_bounds - context_lower_bounds) / 2
        context_mean = context_lower_bounds + context_width

        s2 = ExperimentSpec(n_iter=400,
                            init_dist=KLJoint(context_lower_bounds, context_upper_bounds,
                                              context_mean, np.diag(np.square(context_width)),
                                              theta_lower_bounds, theta_upper_bounds,
                                              np.zeros(theta_dim), np.diag(np.ones(theta_dim) * 4),
                                              PolynomialFeatures(1, bias=True),
                                              0.5),
                            target_dist=Gaussian(context_lower_bounds, context_upper_bounds,
                                                 np.array([0.08, 0.08]), np.diag(np.array([3e-6, 3e-6]))),
                            value_features=RadialBasisFunctions((8, 8),
                                                                [b for b in zip(context_lower_bounds,
                                                                                context_upper_bounds)],
                                                                bias=True, kernel_widths=[800., 800.]),
                            init_policy=KLWLRPolicy(copy.deepcopy(theta_lower_bounds),
                                                    copy.deepcopy(theta_upper_bounds),
                                                    np.zeros(theta_dim), np.diag(np.ones(theta_dim) * 4),
                                                    PolynomialFeatures(1, bias=True),
                                                    0.5),
                            eta=0.5,
                            alpha_schedule=PercentageSchedule(0.15, offset=90, max_alpha=5),
                            n_samples=50,
                            lower_variance_bound=np.array([3e-5, 3e-5]),
                            kl_div_thresh=20.,
                            goal_gan_spec=GoalGANSpec(state_noise_level=0.1, state_distance_threshold=0.01,
                                                      buffer_size=10, p_old_samples=0.2, p_context_samples=0.4,
                                                      gan_pre_train_iters=2000),
                            sagg_riac_spec=SAGGRiacSpec(max_history=80, max_goals=300))

        super(ReachAvoid, self).__init__(name, n_cores, "default", {"default": s2}, ReachAvoidCostFunction)
