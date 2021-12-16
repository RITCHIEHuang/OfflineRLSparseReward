import numpy as np
from gym import spaces
from recsim.simulator import environment
from recsim.simulator.recsim_gym import RecSimGymEnv

from rec_env.offline_env import OfflineEnv
from rec_env.recs_env import create_env, reward_fn


class ObservationAdapter(object):
    """An adapter to convert between user/doc observation and images."""

    def __init__(self, input_observation_space, encode_format="vec"):
        self._input_observation_space = input_observation_space
        self._encode_format = encode_format
        user_space = input_observation_space.spaces["user"]
        doc_space = input_observation_space.spaces["doc"]
        self._num_candidates = len(doc_space.spaces)

        doc_space_shape = spaces.flatdim(list(doc_space.spaces.values())[0])
        # Use the longer of user_space and doc_space as the shape of each row.
        if encode_format == "img":
            obs_shape = (
                np.max([spaces.flatdim(user_space), doc_space_shape]),
            )
            self._observation_shape = (self._num_candidates + 1,) + obs_shape
        else:
            self._observation_shape = [
                spaces.flatdim(user_space)
                + doc_space_shape * self._num_candidates
            ]

        self._observation_dtype = user_space.dtype

    # Pads an array with zeros to make its shape identical to  _observation_shape.
    def _pad_with_zeros(self, array):
        width = self._observation_shape[-1] - len(array)
        return np.pad(array, (0, width), mode="constant")

    @property
    def output_observation_space(self):
        """The output observation space of the adapter."""
        user_space = self._input_observation_space.spaces["user"]
        doc_space = self._input_observation_space.spaces["doc"]
        user_dim = spaces.flatdim(user_space)

        if self._encode_format == "img":
            low = np.concatenate(
                [
                    self._pad_with_zeros(np.ones(user_dim) * -np.inf).reshape(
                        1, -1
                    )
                ]
                + [
                    self._pad_with_zeros(
                        np.ones(spaces.flatdim(d)) * -np.inf
                    ).reshape(1, -1)
                    for d in doc_space.spaces.values()
                ]
            )
            high = np.concatenate(
                [
                    self._pad_with_zeros(np.ones(user_dim) * np.inf).reshape(
                        1, -1
                    )
                ]
                + [
                    self._pad_with_zeros(
                        np.ones(spaces.flatdim(d)) * np.inf
                    ).reshape(1, -1)
                    for d in doc_space.spaces.values()
                ]
            )
        else:
            low = np.ones(self._observation_shape[-1]) * -np.inf
            high = np.ones(self._observation_shape[-1]) * np.inf
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def encode(self, observation):
        """Encode user observation and document observations to specific format."""
        if self._encode_format == "vec":
            return self._vec_encode(observation)
        else:
            return self._img_encode(observation)

    def _vec_encode(self, observation):
        """Encode user observation and document observations to a vector."""
        # It converts the observation from the simulator to a numpy array to be
        # consumed by agent, which assume the input is a vector Concatenating user's observation and documents' observation.

        doc_space = zip(
            self._input_observation_space.spaces["doc"].spaces.values(),
            observation["doc"].values(),
        )
        vec = np.concatenate(
            [
                spaces.flatten(
                    self._input_observation_space.spaces["user"],
                    observation["user"],
                )
            ]
            + [spaces.flatten(doc_space, d) for doc_space, d in doc_space]
        )

        return vec

    def _img_encode(self, observation):
        """Encode user observation and document observations to an image."""
        # It converts the observation from the simulator to a numpy array to be
        # consumed by DQN agent, which assume the input is a "image".
        # The first row is user's observation. The remaining rows are documents'
        # observation, one row for each document.
        image = np.zeros(
            self._observation_shape + (1,),
            dtype=self._observation_dtype,
        )
        image[0, :, 0] = self._pad_with_zeros(
            spaces.flatten(
                self._input_observation_space.spaces["user"],
                observation["user"],
            )
        )
        doc_space = zip(
            self._input_observation_space.spaces["doc"].spaces.values(),
            observation["doc"].values(),
        )
        image[1:, :, 0] = np.array(
            [
                self._pad_with_zeros(spaces.flatten(doc_space, d))
                for doc_space, d in doc_space
            ]
        )

        return image


class RecsEnv(RecSimGymEnv, OfflineEnv):
    def __init__(
        self,
        raw_environment,
        reward_aggregator,
        obs_encode_format="vec",
        **kwargs,
    ):
        RecSimGymEnv.__init__(
            self,
            raw_environment,
            reward_aggregator,
        )
        OfflineEnv.__init__(self, **kwargs)

        self._obs_adapter = ObservationAdapter(
            self.raw_observation_space, obs_encode_format
        )

    @property
    def raw_observation_space(self):
        """Returns the observation space of the environment.

        Each observation is a dictionary with three keys `user`, `doc` and
        `response` that includes observation about user state, document and user
        response, respectively.
        """
        if isinstance(self._environment, environment.MultiUserEnvironment):
            user_obs_space = self._environment.user_model[
                0
            ].observation_space()
            resp_obs_space = self._environment.user_model[0].response_space()
            user_obs_space = spaces.Tuple(
                [user_obs_space] * self._environment.num_users
            )
            resp_obs_space = spaces.Tuple(
                [resp_obs_space] * self._environment.num_users
            )

        if isinstance(self._environment, environment.SingleUserEnvironment):
            user_obs_space = self._environment.user_model.observation_space()
            resp_obs_space = self._environment.user_model.response_space()

        return spaces.Dict(
            {
                "user": user_obs_space,
                "doc": self._environment.candidate_set.observation_space(),
                "response": resp_obs_space,
            }
        )

    @property
    def observation_space(self):
        return self._obs_adapter.output_observation_space

    @property
    def action_space(self):
        """Returns the action space of the environment.

        Each action is a vector that specified document slate. Each element in the
        vector corresponds to the index of the document in the candidate set.
        """
        action_space = spaces.Discrete(
            self._environment.num_candidates
            # * np.ones((self._environment.slate_size,))
        )
        return action_space

    def step(self, action):
        """Runs one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling `reset()`
        to reset this environment's state. Accepts an action and returns a tuple
        (observation, reward, done, info).

        Args:
        action (object): An action provided by the environment

        Returns:
        A four-tuple of (observation, reward, done, info) where:
            observation (object): agent's observation that include
            1. User's state features
            2. Document's observation
            3. Observation about user's slate responses.
            reward (float) : The amount of reward returned after previous action
            done (boolean): Whether the episode has ended, in which case further
            step() calls will return undefined results
            info (dict): Contains responses for the full slate for
            debugging/learning.
        """
        user_obs, doc_obs, responses, done = self._environment.step(action)
        if isinstance(self._environment, environment.MultiUserEnvironment):
            all_responses = tuple(
                tuple(
                    response.create_observation()
                    for response in single_user_resps
                )
                for single_user_resps in responses
            )
        else:  # single user environment
            all_responses = tuple(
                response.create_observation() for response in responses
            )
        obs = dict(user=user_obs, doc=doc_obs, response=all_responses)

        # extract rewards from responses
        reward = self._reward_aggregator(responses)
        info = self.extract_env_info()
        info["raw_obs"] = obs
        encoded_obs = self._obs_adapter.encode(obs)
        return encoded_obs, reward, done, info

    def reset(self, return_raw=False):
        user_obs, doc_obs = self._environment.reset()
        obs = dict(user=user_obs, doc=doc_obs, response=None)
        encoded_obs = self._obs_adapter.encode(obs)

        if return_raw:
            return encoded_obs, {"raw_obs": obs}
        else:
            return encoded_obs


def get_recs_env(**kwargs):
    env = create_env()
    recs_env = RecsEnv(env, reward_fn, **kwargs)
    return recs_env
