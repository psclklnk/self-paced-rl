from gym.envs.registration import register
import numpy as np

register(
    id='FetchReachAvoid-v1',
    entry_point='sprl.util.gym_envs.reach_avoid:FetchReachAvoidEnv',
    max_episode_steps=150,
)

register(
    id='FetchReachAvoidSB-v1',
    entry_point='sprl.util.gym_envs.reach_avoid_sb:FetchReachAvoidStepBasedEnv',
    max_episode_steps=150,
    kwargs={'reward_type': 'dense', 'fixed_context': np.array([0.08, 0.08])}
)

register(
    id='FetchReachAvoidSBEasy-v1',
    entry_point='sprl.util.gym_envs.reach_avoid_sb:FetchReachAvoidStepBasedEnv',
    max_episode_steps=150,
    kwargs={'reward_type': 'dense', 'fixed_context': np.array([0.01, 0.01])}
)
