from gym.envs.registration import register
from d4rl.habitat import gym_envs
from d4rl import infos


# Single Policy datasets
register(
    id='habitat-v0',
    entry_point='d4rl.habitat.habitat_env:get_habitat_env',
    max_episode_steps=500,
    kwargs={
        'deprecated': False,
        'ref_min_score': 0,
        'ref_max_score': 1,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_medium.hdf5'
    }
)
