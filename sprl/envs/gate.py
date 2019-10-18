from sprl.envs.abstractenv import AbstractEnvironment
from sprl.envs.spec import ExperimentSpec, GoalGANSpec, SAGGRiacSpec
from sprl.util.schedule import PercentageSchedule
from sprl.distributions.gaussian import Gaussian
from sprl.features.rbf import RadialBasisFunctions
from sprl.features.polynomial import PolynomialFeatures
from sprl.distributions.kl_joint import KLJoint
from sprl.policies.kl_wlr import KLWLRPolicy
import numpy as np
import copy


class GateCostFunction:

    def __init__(self, pid):
        self.pid = pid

    def __call__(self, context, theta):
        s, a = self._run_experiment(context, theta)
        r = 10 * np.exp(-np.minimum(300.0, np.linalg.norm(s[-1, :]))) - 1e-4 * np.sum(np.linalg.norm(a, axis=1) ** 2)
        return 100 + np.maximum(-100, r), (np.linalg.norm(s[-1, :]) < 5e-2).astype(np.float)

    def set_seed(self, seed):
        print("Process " + str(self.pid) + " - Setting seed " + str(seed))
        np.random.seed(seed)

    @staticmethod
    def _run_experiment(context, theta):
        gate_pos = context[0]
        gate_spread = context[1]

        ck_1 = np.reshape(theta[0:4], (2, 2))
        k_1 = np.reshape(theta[4:6], (2,))
        goal_1 = np.concatenate(([theta[6]], [2.5]))

        ck_2 = np.reshape(theta[7:11], (2, 2))
        k_2 = np.reshape(theta[11:13], (2,))
        goal_2 = np.concatenate(([theta[13]], [0]))

        dt = 0.05
        offset = np.array([5, -1])
        peturbations = np.random.normal(0, 0.05, (100, 2))

        controller2 = False
        states = [np.array([0, 5], dtype=np.float)]
        actions = []
        for i in range(0, 100):
            cur_state = states[-1]
            if controller2 or cur_state[1] - goal_1[1] < 1e-2:
                controller2 = True
                action = np.dot(ck_2, goal_2 - cur_state) + k_2
            else:
                action = np.dot(ck_1, goal_1 - cur_state) + k_1
            actions.append(action)

            new_state = cur_state + dt * (action + offset + peturbations[i, :])

            if cur_state[1] >= 2.5 > new_state[1]:
                alpha = (2.5 - cur_state[1]) / (new_state[1] - cur_state[1])
                x_crit = alpha * new_state[0] + (1 - alpha) * cur_state[0]

                if np.abs(x_crit - gate_pos) > gate_spread:
                    states.append(np.array([x_crit, 2.5]))
                    break

            states.append(new_state)

        return np.array(states), np.array(actions)


class Gate(AbstractEnvironment):

    def __init__(self, name, n_cores):
        theta_dim = 16
        theta_lower_bounds = np.ones(theta_dim) * -10
        theta_upper_bounds = np.ones(theta_dim) * 10
        dist = (theta_upper_bounds - theta_lower_bounds) / 2
        mean = dist + theta_lower_bounds
        var = np.diag(np.square(dist))

        s3 = ExperimentSpec(n_iter=350,
                            init_dist=KLJoint(np.array([-5, 0.1]), np.array([5, 0.5]),
                                              np.array([0, 0.25]), np.array([[7, 0], [0, 0.003]]),
                                              copy.deepcopy(theta_lower_bounds), copy.deepcopy(theta_upper_bounds),
                                              copy.deepcopy(mean), copy.deepcopy(var),
                                              PolynomialFeatures(1, bias=True), 0.4),
                            target_dist=Gaussian(np.array([-5, 0.1]), np.array([5, 0.5]),
                                                 np.array([-4, 0.2]),
                                                 np.array([[0.1, 0], [0, 0.001]])),
                            value_features=RadialBasisFunctions((10, 5),
                                                                [b for b in zip([-5, 0.1], [5, 0.5])],
                                                                bias=True, kernel_widths=np.array([0.6, 150.])),
                            init_policy=KLWLRPolicy(copy.deepcopy(theta_lower_bounds),
                                                    copy.deepcopy(theta_upper_bounds),
                                                    copy.deepcopy(mean), copy.deepcopy(var),
                                                    PolynomialFeatures(1, bias=True), 0.4),
                            eta=0.4,
                            alpha_schedule=PercentageSchedule(0.02, offset=140, max_alpha=2.5),
                            n_samples=100,
                            goal_gan_spec=GoalGANSpec(state_noise_level=0.05, state_distance_threshold=0.01,
                                                      buffer_size=10, p_old_samples=0.2, p_context_samples=0.4,
                                                      gan_train_interval=3),
                            sagg_riac_spec=SAGGRiacSpec(max_history=100, max_goals=200))

        s4 = ExperimentSpec(n_iter=600,
                            init_dist=KLJoint(np.array([-5, 0.1]), np.array([5, 0.5]),
                                              np.array([0, 0.25]), np.array([[7, 0], [0, 0.003]]),
                                              copy.deepcopy(theta_lower_bounds), copy.deepcopy(theta_upper_bounds),
                                              copy.deepcopy(mean), copy.deepcopy(var),
                                              PolynomialFeatures(1, bias=True), 0.25),
                            target_dist=Gaussian(np.array([-5, 0.1]), np.array([5, 0.5]),
                                                 np.array([0, 0.25]), np.array([[7, 0], [0, 0.003]])),
                            value_features=RadialBasisFunctions((10, 5),
                                                                [b for b in zip([-5, 0.1], [5, 1])],
                                                                bias=True, kernel_widths=np.array([0.6, 150.])),
                            init_policy=KLWLRPolicy(copy.deepcopy(theta_lower_bounds),
                                                    copy.deepcopy(theta_upper_bounds),
                                                    copy.deepcopy(mean), copy.deepcopy(var),
                                                    PolynomialFeatures(1, bias=True), 0.25),
                            eta=0.25,
                            alpha_schedule=PercentageSchedule(0.002, offset=140, max_alpha=1e6),
                            n_samples=100,
                            goal_gan_spec=GoalGANSpec(state_noise_level=0.05, state_distance_threshold=0.01,
                                                      buffer_size=10, p_old_samples=0.2, p_context_samples=0.4),
                            sagg_riac_spec=SAGGRiacSpec(max_history=100, max_goals=500))

        super(Gate, self).__init__(name, n_cores, "precision", {"precision": s3, "global": s4}, GateCostFunction)
