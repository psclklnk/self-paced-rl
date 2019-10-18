import os
from gym import utils as gym_utils
from gym.envs.robotics import fetch_env, utils
import numpy as np
from gym import spaces
from gym.envs.robotics.fetch_env import goal_distance

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'fetch', 'reach_avoid.xml')


class FetchReachAvoidStepBasedEnv(fetch_env.FetchEnv, gym_utils.EzPickle):
    def __init__(self, reward_type='sparse', fixed_context=None):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }

        self.obstacle_body_id = None
        self.obstacle_geom_id = None
        self.obstacle_information = None
        self.collision_detected = False
        self.fixed_context = fixed_context
        self.goal_reached = False

        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=5,  # n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)

        self.obstacle_lo = np.array([0.02, 0.02])
        self.obstacle_hi = np.array([0.14, 0.14])
        self.obstacle_geom_id = self.sim.model._geom_name2id["obstacle_geom"]
        self.obstacle_body_id = self.sim.model._body_name2id["obstacle"]

        self.obstacle2_geom_id = self.sim.model._geom_name2id["obstacle_geom1"]
        self.obstacle2_body_id = self.sim.model._body_name2id["obstacle1"]

        self.fixed_obstacle_geom_ids = [self.sim.model._geom_name2id["obstacle_geom2"],
                                        self.sim.model._geom_name2id["obstacle_geom3"]]
        self._sample_goal()

        if self.reward_type == "sparse":
            obs = self.postprocess_action(self._get_obs())
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
            ))
        else:
            self.observation_space = spaces.Box(np.array([1.1, -1, 0.35, -1]), np.array([1.5, 1, 1.15, 1]),
                                                dtype='float32')

        self.original_action_space = self.action_space
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype='float32')

        gym_utils.EzPickle.__init__(self)

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        for coni in range(0, self.sim.data.ncon):
            con = self.sim.data.contact[coni]

            block_collision = con.geom1 == self.obstacle_geom_id or con.geom2 == self.obstacle_geom_id
            block2_collision = con.geom1 == self.obstacle2_geom_id or con.geom2 == self.obstacle2_geom_id
            rem_collision = con.geom1 in self.fixed_obstacle_geom_ids or con.geom2 in self.fixed_obstacle_geom_ids

            # This means we have a collision with one of the obstacles
            if rem_collision or block_collision or block2_collision:
                # Now we check if we have a collision with the robot
                if self.sim.model._geom_id2name[con.geom1].startswith("robot") or \
                        self.sim.model._geom_id2name[con.geom2].startswith("robot"):
                    self.collision_detected = True

        super(FetchReachAvoidStepBasedEnv, self)._step_callback()

    def postprocess_action(self, obs):
        if self.reward_type == "sparse":
            obs["observation"] = np.array([obs["observation"][0], obs["observation"][5], obs["observation"][1],
                                           obs["observation"][6]])
            obs["achieved_goal"] = obs["achieved_goal"][0:2]
            obs["desired_goal"] = obs["desired_goal"][0:2]
            return obs
        else:
            return np.array(
                [obs["observation"][0], obs["observation"][5], obs["observation"][1], obs["observation"][6]])

    def _sample_goal(self):
        goal = np.array([1.3, 0.35, 0.5])

        if self.obstacle_body_id is not None:
            new_context = np.copy(self.fixed_context) \
                if self.fixed_context is not None else np.random.uniform(self.obstacle_lo, self.obstacle_hi)
            self._set_obstacle_information(new_context)

        return goal.copy()

    def _set_obstacle_information(self, obs_inf):
        self.obstacle_information = obs_inf

        size = np.copy(self.sim.model.geom_size[self.obstacle_geom_id])
        size[0] = self.obstacle_information[0]

        xp = np.copy(self.sim.data.body_xpos[self.obstacle_body_id])
        correction_x = np.cos(-0.5) * size[0]
        correction_y = np.sin(-0.5) * size[0]
        xp[0] = 1.14 + correction_x
        xp[1] = 0.92 + correction_y
        # xp[0] = 1.12 + size[0]
        self.sim.model.body_pos[self.obstacle_body_id][:] = xp

        self.sim.model.geom_rbound[self.obstacle_geom_id] = np.sqrt(np.sum(np.square(size)))
        self.sim.model.geom_size[self.obstacle_geom_id][:] = size
        self.sim.data.body_xpos[self.obstacle_body_id][:] = xp

        # The same for the second block
        size = np.copy(self.sim.model.geom_size[self.obstacle2_geom_id])
        size[0] = self.obstacle_information[1]

        xp = np.copy(self.sim.data.body_xpos[self.obstacle2_body_id])
        correction_x = np.cos(-0.2) * size[0]
        correction_y = np.sin(-0.2) * size[0]
        xp[0] = 1.47 - correction_x
        xp[1] = 0.65 - correction_y
        # xp[0] = 1.48 - size[0]
        self.sim.model.body_pos[self.obstacle2_body_id][:] = xp

        self.sim.model.geom_rbound[self.obstacle2_geom_id] = np.sqrt(np.sum(np.square(size)))
        self.sim.model.geom_size[self.obstacle2_geom_id][:] = size
        self.sim.data.body_xpos[self.obstacle2_body_id][:] = xp

    def _is_success(self, achieved_goal, desired_goal):
        suc = super(FetchReachAvoidStepBasedEnv, self)._is_success(achieved_goal, desired_goal)
        return suc and (not self.collision_detected)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498 - 0.0418, 0.005 + 0.4009, -0.431 - 0.055 + self.gripper_extra_height]) + \
                         self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def reset(self):
        # Copied from robot_env (the goal_env checks raise an exception for this)
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()

        self.collision_detected = False
        self.goal_reached = False
        return self.postprocess_action(obs)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            if self.collision_detected:
                return -1
            else:
                return -(d > self.distance_threshold).astype(np.float32)
        else:
            return np.exp(-2.5 * d)

    def step(self, action):
        if self.collision_detected:
            obs_old = self._get_obs()
            info = {
                'is_success': self._is_success(obs_old['achieved_goal'], self.goal),
                'is_collision': self.collision_detected
            }
            reward = self.compute_reward(obs_old['achieved_goal'], self.goal, info)

            return self.reset(), reward, False, info
        else:
            clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
            cur_z = self._get_obs()["achieved_goal"][2]
            complete_action = np.concatenate((clipped_action, [2 * (0.5 - cur_z), 0]))
            tmp = self.action_space
            self.action_space = self.original_action_space
            obs, reward, __, info = super(FetchReachAvoidStepBasedEnv, self).step(complete_action)
            self.action_space = tmp
            self.goal_reached = info["is_success"]
            info['is_collision'] = self.collision_detected
            return self.postprocess_action(obs), reward, False, info
