from typing import Dict

import numpy as np
from gym import spaces

from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.preprocessing import get_obs_shape


class ObsDictWrapper(VecEnvWrapper):
    """
    Wrapper for a VecEnv which overrides the observation space for
    Hindsight Experience Replay to support dict observations.

    :param env: The vectorized environment to wrap.
    """

    def __init__(self, venv: VecEnv):
        super(ObsDictWrapper, self).__init__(venv, venv.observation_space, venv.action_space)

        self.venv = venv

        self.spaces = list(venv.observation_space.spaces.values())

        # assume all spaces have identical type
        assert all(isinstance(sp, type(self.spaces[0])) for sp in self.spaces)

        # get dimensions of observation and goal
        self.obs_dim = get_obs_shape(venv.observation_space.spaces["observation"])
        self.goal_dim = get_obs_shape(venv.observation_space.spaces["achieved_goal"])

        # new observation space with concatenated observation and (desired) goal
        # for the different types of spaces
        if isinstance(self.spaces[0], spaces.Box):
            self.observation_space = spaces.Box(
                np.concatenate([
                    venv.observation_space.spaces["observation"].low,
                    venv.observation_space.spaces["desired_goal"].low,
                ], axis=-1),
                np.concatenate([
                    venv.observation_space.spaces["observation"].high,
                    venv.observation_space.spaces["desired_goal"].high,
                ], axis=-1), dtype=np.float32)

        elif isinstance(self.spaces[0], spaces.MultiBinary):
            assert self.obs_dim[:-1] == self.goal_dim[:-1]
            self.observation_space = spaces.MultiBinary((
                *self.obs_dim[:-1], self.obs_dim[-1] + self.goal_dim[-1]
            ))

        elif isinstance(self.spaces[0], spaces.Discrete):
            self.observation_space = spaces.MultiDiscrete([
                venv.observation_space.spaces["observation"].n,
                venv.observation_space.spaces["desired_goal"].n,
            ])

        else:
            raise NotImplementedError(f"{type(self.spaces[0])} space is not supported")

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()

    @staticmethod
    def convert_dict(
        observation_dict: Dict[str, np.ndarray], observation_key: str = "observation", goal_key: str = "desired_goal"
    ) -> np.ndarray:
        """
        Concatenate observation and (desired) goal of observation dict.

        :param observation_dict: Dictionary with observation.
        :param observation_key: Key of observation in dictionary.
        :param goal_key: Key of (desired) goal in dictionary.
        :return: Concatenated observation.
        """
        return np.concatenate([
            observation_dict[observation_key],
            observation_dict[goal_key],
        ], axis=-1)
