from sprl.envs.abstractenv import AbstractEnvironment
from sprl.envs.spec import ExperimentSpec
from sprl.util.schedule import PercentageSchedule
from sprl.distributions.gaussian import Gaussian
from sprl.features.rbf import RadialBasisFunctions
from sprl.features.polynomial import PolynomialFeatures
from sprl.distributions.kl_joint import KLJoint
from sprl.policies.kl_wlr import KLWLRPolicy
import os
import numpy as np
import mujoco_py
from mujoco_py import functions as mj_fun
import copy


class PointMassCostFunction:

    def __init__(self, pid):
        self.pid = pid
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xml", "point-mass.xml")
        self.sim = mujoco_py.MjSim(mujoco_py.load_model_from_path(path))
        self.sparse = False

    def __call__(self, cs, xs, render=False):
        distances, action_costs, success = self._run_experiment(cs, xs, render=render)

        if self.sparse:
            reward = success
        else:
            reward = np.exp(-2 * distances[-1])

        return reward, success

    def set_seed(self, seed):
        print("Process " + str(self.pid) + " - Setting seed " + str(seed))
        np.random.seed(seed)

    def _run_experiment(self, context, theta, render=False):
        # Reset the environment
        mj_fun.mj_resetData(self.sim.model, self.sim.data)
        ball_id = self.sim.model._body_name2id["ball"]
        target_id = self.sim.model._body_name2id["target"]

        # Set the position of the goal
        self.sim.model.body_pos[target_id, 0:2] = context
        self.sim.data.body_xpos[target_id, 0:2] = context

        if render:
            viewer = mujoco_py.MjViewer(self.sim)
        else:
            viewer = None

        target_pos = self.sim.data.body_xpos[target_id][0:2]
        k = 0
        torques = []
        distances = []
        while k < 400:
            ball_pos = self.sim.data.body_xpos[ball_id][0:2]
            ball_vel = self.sim.data.body_xvelp[ball_id][0:2]
            distances.append(np.linalg.norm(target_pos - ball_pos))

            # Compute the controls
            trq = (theta - ball_pos) + np.sqrt(2) * (np.zeros(2) - ball_vel)
            torques.append(trq)

            # Advance the simulation
            self.sim.data.ctrl[:] = trq
            self.sim.step()

            k += 1

            if viewer is not None:
                viewer.render()

        return distances, np.mean(np.sum(np.square(torques), axis=1), axis=0), 1. if distances[-1] < 0.05 else 0.


class PointMass(AbstractEnvironment):

    def __init__(self, name, n_cores):
        context_lower_bounds = np.array([-1.5, -1.5])
        context_upper_bounds = np.array([1.5, 1.5])

        context_mean = np.array([-1.3, 1.3])
        context_width = np.array([0.2, 0.2])

        theta_dim = 2
        mean = np.zeros((theta_dim,))
        cov = np.diag(2 * np.ones((theta_dim,)))

        policy_lower_bounds = np.ones(theta_dim) * -10
        policy_upper_bounds = np.ones(theta_dim) * 10

        s1 = ExperimentSpec(n_iter=250,
                            init_dist=KLJoint(copy.deepcopy(context_lower_bounds),
                                              copy.deepcopy(context_upper_bounds),
                                              copy.deepcopy(context_mean),
                                              copy.deepcopy(np.diag(np.square(context_width))),
                                              policy_lower_bounds,
                                              policy_upper_bounds,
                                              copy.deepcopy(mean), copy.deepcopy(cov),
                                              PolynomialFeatures(order=1, bias=True),
                                              epsilon=0.5, max_eta=10.),
                            target_dist=Gaussian(context_lower_bounds, context_upper_bounds,
                                                 np.array([1.3, -1.3]), np.diag([1e-2, 1e-2])),
                            value_features=RadialBasisFunctions((5, 5),
                                                                [b for b in zip(context_lower_bounds,
                                                                                context_upper_bounds)],
                                                                bias=True, kernel_widths=[3.]),
                            init_policy=KLWLRPolicy(policy_lower_bounds,
                                                    policy_upper_bounds,
                                                    np.copy(mean), np.copy(cov),
                                                    PolynomialFeatures(order=1, bias=True), epsilon=0.5, max_eta=10.),
                            eta=0.5,
                            alpha_schedule=PercentageSchedule(0.75, offset=45, max_alpha=10),
                            buffer_size=10,
                            n_samples=50,
                            lower_variance_bound=0.04,
                            kl_div_thresh=10.,
                            # This is just an experiment for showing non-linear movement of the context distribution.
                            # It is not meant to be run with the other algorithms
                            goal_gan_spec=None,
                            sagg_riac_spec=None)

        super(PointMass, self).__init__(name, n_cores, "default", {"default": s1}, PointMassCostFunction)
