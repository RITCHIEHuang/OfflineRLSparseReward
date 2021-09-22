import numpy as np
from scipy import stats
from scipy.special import expit

from gym import spaces
import matplotlib.pyplot as plt

from recsim import document
from recsim import user
from recsim.choice_model import MultinomialLogitChoiceModel

from recsim.simulator import environment, recsim_gym

from envs.rec_env.configs import (
    MAX_HIST_CLICK_TAG_NUM,
    SCORE_WEIGHT_MATRIX,
    STATIC_FEATURE_DIM,
    DYNAMIC_FEATURE_DIM,
    MIN_VIDEO_DURATION,
    MAX_VIDEO_DURATION,
    EMB_DIM,
    MAX_NEXT_TIME,
    MAX_CUM_NEXT_TIME,
    SEED,
)


class Video(document.AbstractDocument):
    def __init__(self, v_id, emb, duration, ctr):
        self.emb = emb
        self.duration = duration
        self.ctr = ctr

        super(Video, self).__init__(v_id)

    def create_observation(self):
        return {
            "emb": self.emb,
            "duration": np.array([self.duration]),
            "ctr": np.array([self.ctr]),
        }

    @staticmethod
    def observation_space():
        return spaces.Dict(
            {
                # tag
                "emb": spaces.Box(
                    shape=(EMB_DIM,), dtype=np.float32, low=-1.0, high=1.0
                ),
                # duration min
                "duration": spaces.Box(
                    shape=(1,),
                    dtype=np.float32,
                    low=1.0,
                    high=60.0,
                ),
                # ctr
                "ctr": spaces.Box(
                    shape=(1,),
                    dtype=np.float32,
                    low=0.0,
                    high=1.0,
                ),
            }
        )

    def __str__(self):
        return "Video {} with emb {}, duration {} min, ctr {}".format(
            self._doc_id, self.emb, self.duration, self.ctr
        )


class VideoSampler(document.AbstractDocumentSampler):
    def __init__(
        self,
        doc_ctor=Video,
        min_emb_value=-1.0,
        max_emb_value=1.0,
        video_duration_mean=2.0,
        video_duration_std=3.0,
        **kwargs,
    ):
        super(VideoSampler, self).__init__(doc_ctor, **kwargs)
        self._video_count = 0
        self._min_emb_value = min_emb_value
        self._max_emb_value = max_emb_value
        self._video_duration_mean = video_duration_mean
        self._video_duration_std = video_duration_std

    def sample_document(self):
        doc_features = {}
        doc_features["v_id"] = self._video_count
        doc_features["emb"] = self._rng.uniform(
            self._min_emb_value,
            self._max_emb_value,
            EMB_DIM,
        )
        doc_features["duration"] = np.clip(
            self._rng.normal(
                self._video_duration_mean, self._video_duration_std
            ),
            MIN_VIDEO_DURATION,
            MAX_VIDEO_DURATION,
        )
        doc_features["ctr"] = self._rng.random_sample()
        self._video_count += 1
        return self._doc_ctor(**doc_features)


class UserState(user.AbstractUserState):
    def __init__(
        self,
        static_feature,
        click_list,
        dynamic_feature,
        cum_next_time,
        score_tau,
        next_time_lambda,
    ):

        self.static_feature = static_feature
        self.dynamic_feature = dynamic_feature
        self.click_list = click_list
        self.cum_next_time = cum_next_time
        self.day = cum_next_time // 1440
        self.score_tau = score_tau
        self.next_time_lambda = next_time_lambda

    def create_observation(self):
        return {
            "static_feature": self.static_feature,
            "dynamic_feature": self.dynamic_feature,
            "cum_next_time": self.cum_next_time,
            "day": self.day,
        }

    @staticmethod
    def observation_space():
        return spaces.Dict(
            {
                "cum_next_time": spaces.Box(
                    shape=tuple(),
                    dtype=np.float32,
                    low=0.0,
                    high=MAX_CUM_NEXT_TIME,
                ),
                "day": spaces.Box(
                    shape=tuple(), dtype=np.int64, low=0, high=10
                ),
                "static_feature": spaces.Box(
                    shape=(STATIC_FEATURE_DIM,),
                    dtype=np.float32,
                    low=-1.0,
                    high=1.0,
                ),
                "dynamic_feature": spaces.Box(
                    shape=(DYNAMIC_FEATURE_DIM,),
                    dtype=np.float32,
                    low=-1.0,
                    high=1.0,
                ),
            }
        )

    def score_document(self, doc_obs):
        return expit(
            (
                (
                    SCORE_WEIGHT_MATRIX
                    @ np.concatenate(
                        [self.static_feature, self.dynamic_feature], axis=0
                    )
                )
                @ doc_obs["emb"]
            )
            / self.score_tau
        )


class UserSampler(user.AbstractUserSampler):
    def __init__(
        self,
        user_ctor=UserState,
        cum_next_time=0.0,
        click_list=[],
        min_static_feature=-1.0,
        max_static_feature=1.0,
        score_tau_mean=0.0,
        score_tau_std=1.0,
        next_time_lambda=800,
        **kwargs,
    ):
        self.cum_next_time = cum_next_time
        self.click_list = click_list
        self._min_static_feature = min_static_feature
        self._max_static_feature = max_static_feature
        self._next_time_lambda = next_time_lambda
        self._score_tau_mean = score_tau_mean
        self._score_tau_std = score_tau_std
        super(UserSampler, self).__init__(user_ctor, **kwargs)

    def sample_user(self):
        state_parameters = {
            "click_list": self.click_list,
            "cum_next_time": self.cum_next_time,
            "static_feature": self._rng.uniform(
                self._min_static_feature,
                self._max_static_feature,
                STATIC_FEATURE_DIM,
            ),
            "dynamic_feature": np.zeros(DYNAMIC_FEATURE_DIM),
            "score_tau": self._rng.normal(
                self._score_tau_mean, self._score_tau_std
            ),
            "next_time_lambda": self._next_time_lambda,
        }
        return self._user_ctor(**state_parameters)


class UserResponse(user.AbstractResponse):
    def __init__(self, clicked=False, next_time=0.0, reward=0.0):
        self.clicked = clicked
        self.next_time = next_time
        self.reward = reward

    def create_observation(self):
        return {
            "click": int(self.clicked),
            "next_time": np.array(self.next_time),
            "reward": np.array(self.reward),
        }

    @classmethod
    def response_space(cls):
        return spaces.Dict(
            {
                "click": spaces.Discrete(2),
                "next_time": spaces.Box(
                    low=0.0,
                    high=MAX_NEXT_TIME,
                    shape=tuple(),
                    dtype=np.float32,
                ),
                "reward": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=tuple(),
                    dtype=np.float32,
                ),
            }
        )


class UserModel(user.AbstractUserModel):
    def __init__(self, slate_size, seed=0):
        super(UserModel, self).__init__(
            UserResponse, UserSampler(UserState, seed=seed), slate_size
        )

        self.choice_model = MultinomialLogitChoiceModel({})

    def simulate_response(self, slate_documents):
        # List of empty responses
        responses = [self._response_model_ctor() for _ in slate_documents]
        # Get click from of choice model.
        self.choice_model.score_documents(
            self._user_state,
            [doc.create_observation() for doc in slate_documents],
        )
        scores = self.choice_model.scores
        selected_index = self.choice_model.choose_item()

        assert selected_index is not None
        # Populate clicked item.
        self._generate_response(
            slate_documents[selected_index], responses[selected_index]
        )
        return responses

    def _generate_response(self, doc, response):
        response.clicked = True

        # TODO generate watch time for video
        watch_time = np.random.uniform(0, doc.duration)
        random_next_time = np.random.exponential(
            self._user_state.next_time_lambda
        )
        response.next_time = watch_time + random_next_time

        # reward
        cur_day = self._user_state.cum_next_time // 1440
        new_day = (self._user_state.cum_next_time + response.next_time) // 1440
        if new_day - cur_day == 1:
            response.reward = 1.0
        else:
            response.reward = 0.0

    def update_state(self, slate_documents, responses):
        """Update user state"""
        for doc, response in zip(slate_documents, responses):
            if response.clicked:
                self._user_state.click_list.append(doc)

            self._user_state.cum_next_time += response.next_time

        self._user_state.click_list = self._user_state.click_list[
            -MAX_HIST_CLICK_TAG_NUM:
        ]
        self._user_state.dynamic_feature = np.mean(
            [doc.emb for doc in self._user_state.click_list], axis=0
        )

    def is_terminal(self):
        """Returns a boolean indicating if the session is over."""
        return self._user_state.cum_next_time >= 9 * 24 * 60


def test_document():
    sampler = VideoSampler()
    for i in range(5):
        print(sampler.sample_document())
    d = sampler.sample_document()
    print(
        "Videos have observation space:",
        d.observation_space(),
        "\n" "An example realization is: ",
        d.create_observation(),
    )


def test_user():
    sampler = UserSampler()
    cum_next_time_hist = []
    for i in range(1000):
        sampled_user = sampler.sample_user()
        cum_next_time_hist.append(sampled_user.cum_next_time)
    plt.hist(cum_next_time_hist)
    plt.show()


def reward_fn(responses):
    total_reward = 0.0
    # TODO add reward
    for response in responses:
        total_reward += response.reward
    return total_reward


def create_env():
    slate_size = 1
    num_candidates = 10
    user_model = UserModel(slate_size)
    video_sampler = VideoSampler(seed=SEED)
    cs_env = environment.Environment(
        user_model,
        video_sampler,
        num_candidates,
        slate_size,
        resample_documents=True,
    )

    return recsim_gym.RecSimGymEnv(
        cs_env,
        reward_fn,
    )


if __name__ == "__main__":
    # test_document()
    test_user()
