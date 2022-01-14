import numpy as np
import gym

from rec_env.env import get_recs_env


class OfflineEnv(gym.Env):
    """
    Base class for offline RL envs.

    Args:
        dataset_path: path of the dataset.
    """

    def __init__(
        self,
        dataset_path=None,
        ref_max_score=None,
        ref_min_score=None,
        reward_key="reward",
        **kwargs
    ):
        super(OfflineEnv, self).__init__(**kwargs)
        self.dataset_path = self._dataset_path = dataset_path
        self.ref_max_score = ref_max_score
        self.ref_min_score = ref_min_score
        self.reward_key = reward_key

    def get_normalized_score(self, score):
        if (self.ref_max_score is None) or (self.ref_min_score is None):
            raise ValueError("Reference score not provided for env")
        return (score - self.ref_min_score) / (
            self.ref_max_score - self.ref_min_score
        )

    @property
    def dataset_filepath(self):
        return self.dataset_path

    def get_dataset(self, npz_path=None):
        if npz_path is None and self._dataset_path is None:
            raise ValueError("Offline env not configured with a dataset path.")
        npz_path = self._dataset_path

        data_dict = np.load(npz_path)
        data_dict = {k: data_dict[k] for k in list(data_dict.keys())}
        # Run a few quick sanity checks
        for key in [
            "observations",
            "next_observations",
            "actions",
            "rewards",
            "terminals",
            self.reward_key,
        ]:
            assert key in data_dict, "Dataset is missing key %s" % key
        N_samples = data_dict["observations"].shape[0]
        if self.observation_space.shape is not None:
            assert (
                data_dict["observations"].shape[1:]
                == self.observation_space.shape
            ), "Observation shape does not match env: %s vs %s" % (
                str(data_dict["observations"].shape[1:]),
                str(self.observation_space.shape),
            )
        assert data_dict["actions"].shape == (
            N_samples,
            1,
        ), "Action shape does not match env: %s vs %s" % (
            str(data_dict["actions"].shape),
            str(self.action_space.shape),
        )

        data_dict["rewards"] = data_dict[self.reward_key]
        if data_dict["rewards"].shape == (N_samples, 1):
            data_dict["rewards"] = data_dict["rewards"][:, 0]
        assert data_dict["rewards"].shape == (
            N_samples,
        ), "Reward has wrong shape: %s" % (str(data_dict["rewards"].shape))
        if data_dict["terminals"].shape == (N_samples, 1):
            data_dict["terminals"] = data_dict["terminals"][:, 0]
        assert data_dict["terminals"].shape == (
            N_samples,
        ), "Terminals has wrong shape: %s" % (str(data_dict["rewards"].shape))
        return data_dict


class OfflineEnvWrapper(gym.Wrapper, OfflineEnv):
    """
    Wrapper class for offline RL envs.
    """

    def __init__(self, env, **kwargs):
        gym.Wrapper.__init__(self, env)
        OfflineEnv.__init__(self, **kwargs)

    def reset(self):
        return self.env.reset()


def get_recs_offline_env(**kwargs):
    recs_env = get_recs_env(**kwargs)
    recs_offline_env = OfflineEnvWrapper(recs_env, **kwargs)
    return recs_offline_env
