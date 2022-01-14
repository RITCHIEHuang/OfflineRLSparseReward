from collections import defaultdict
import numpy as np
from gym import spaces

from recsim import document
from recsim import user

from recsim.simulator import environment

env_config = {
    # global
    "seed": 9999,
    "slate_size": 1,
    "candidate_size": 10,
    # video
    "max_video_duration": 1.0,
    "min_video_duration": 60.0,
    "emb_dim": 10,
    # user
    "time_budget": 9 * 24 * 60,
    "max_impress_num": 4000,
    "standard_click_num": 30.0,
    "click_threshold": 0.70,
    "leave_prob": 0.0,
    # "next_time_mean": 600.0,
    "next_time_mean": 24 * 60 * 2.0,  # min
    "alpha_min": 0.001,
    "alpha_max": 0.01,
    "feature_dim": 10,
}


class Video(document.AbstractDocument):
    EMB_DIM = env_config["emb_dim"]
    MIN_VIDEO_DURATION = env_config["min_video_duration"]
    MAX_VIDEO_DURATION = env_config["max_video_duration"]

    def __init__(self, v_id, emb, duration):
        self.emb = emb
        self.duration = duration

        super(Video, self).__init__(v_id)

    def create_observation(self):
        return {
            "emb": self.emb,
            "duration": np.array([self.duration]),
            # "ctr": np.array([self.ctr]),
        }

    @classmethod
    def observation_space(cls):
        return spaces.Dict(
            {
                # tag
                "emb": spaces.Box(
                    shape=(cls.EMB_DIM,), dtype=np.float32, low=0.0, high=1.0
                ),
                # duration min
                "duration": spaces.Box(
                    shape=(1,),
                    dtype=np.float32,
                    low=cls.MIN_VIDEO_DURATION,
                    high=cls.MAX_VIDEO_DURATION,
                ),
            }
        )

    def __str__(self):
        return "Video {} with emb {}, duration {} min".format(
            self._doc_id, self.emb, self.duration
        )


class VideoSampler(document.AbstractDocumentSampler):
    def __init__(
        self,
        doc_ctor=Video,
        **kwargs,
    ):
        super(VideoSampler, self).__init__(doc_ctor, **kwargs)
        self._video_count = 0

    def sample_document(self):
        doc_features = {}
        doc_features["v_id"] = self._video_count
        doc_features["emb"] = self._rng.uniform(
            0,
            1,
            self.get_doc_ctor().EMB_DIM,
        )
        doc_features["duration"] = self._rng.uniform(
            self.get_doc_ctor().MIN_VIDEO_DURATION,
            self.get_doc_ctor().MAX_VIDEO_DURATION,
        )
        self._video_count += 1
        return self._doc_ctor(**doc_features)


class UserState(user.AbstractUserState):
    def __init__(
        self,
        interest,
        next_time_mean,
        alpha=0.001,
    ):

        self.interest = interest
        self.next_time_mean = next_time_mean

        self.alpha = alpha
        self.click_dict = defaultdict(float)
        self.impressed_dict = defaultdict(float)

        self.total_impress_num = 0

        self.cum_next_time = 0.0
        self.last_day = None

    def create_observation(self):
        return {
            "interest": self.interest,
            # "cum_next_time": self.cum_next_time,
            # "day": self.day,
        }

    @property
    def cur_day(self):
        return self.get_day(self.cum_next_time)

    def get_day(self, cum_next_time):
        return int(cum_next_time // 1440)

    @classmethod
    def observation_space(cls):
        return spaces.Dict(
            {
                # "cum_next_time": spaces.Box(
                #     shape=tuple(),
                #     dtype=np.float32,
                #     low=0.0,
                #     high=env_config["time_budget"],
                # ),
                # "day": spaces.Box(
                #     shape=tuple(), dtype=np.int64, low=0, high=10
                # ),
                "interest": spaces.Box(
                    shape=(env_config["feature_dim"],),
                    dtype=np.float32,
                    low=0.0,
                    high=1.0,
                )
            }
        )

    def score_document(self, doc_obs):
        # the user is more likely to click on the video with similar feature
        score = (self.interest @ doc_obs["emb"]) / (
            np.sqrt(
                self.interest @ self.interest * doc_obs["emb"] @ doc_obs["emb"]
                + 1e-8
            )
        )
        return score


class UserSampler(user.AbstractUserSampler):
    def __init__(
        self,
        config,
        user_ctor=UserState,
        **kwargs,
    ):
        self.config = config
        super(UserSampler, self).__init__(user_ctor, **kwargs)

    def sample_user(self):
        alpha = self._rng.uniform(
            self.config["alpha_min"],
            self.config["alpha_max"],
        )
        state_parameters = {
            "alpha": alpha,
            "interest": self._rng.uniform(
                0,
                1,
                env_config["feature_dim"],
            ),
            "next_time_mean": self.config["next_time_mean"],
        }
        return self._user_ctor(**state_parameters)


class UserResponse(user.AbstractResponse):
    def __init__(self, clicked=False, next_time=0.0, retention=0.0):
        self.clicked = clicked
        self.next_time = next_time
        self.retention = retention

    def create_observation(self):
        return {
            "clicked": int(self.clicked),
            "next_time": np.array(self.next_time),
            "retention": int(self.retention),
        }

    @classmethod
    def response_space(cls):
        return spaces.Dict(
            {
                "clicked": spaces.Discrete(2),
                "next_time": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=tuple(),
                    dtype=np.float32,
                ),
                "retention": spaces.Discrete(2),
            }
        )


class UserModel(user.AbstractUserModel):
    def __init__(self, config):
        super(UserModel, self).__init__(
            UserResponse,
            UserSampler(config, UserState, seed=config["seed"]),
            config["slate_size"],
        )
        self.config = config
        # self.choice_model = MultinomialLogitChoiceModel({})

    def simulate_response(self, slate_documents):
        # List of empty responses
        responses = [self._response_model_ctor() for _ in slate_documents]
        # Get click from of choice model.
        doc_obs = [doc.create_observation() for doc in slate_documents]
        # self.choice_model.score_documents(
        #     self._user_state,
        #     doc_obs,
        # )
        # scores = self.choice_model.scores
        # selected_index = self.choice_model.choose_item()

        selected_index = 0
        scores = [self._user_state.score_document(doc) for doc in doc_obs]

        assert selected_index is not None
        # Populate clicked item.
        self._generate_response(
            slate_documents[selected_index],
            responses[selected_index],
            scores[selected_index],
        )
        return responses

    def _generate_response(self, doc, response, score):
        next_time = None
        if score >= self.config["click_threshold"]:
            response.clicked = True
            watch_time = score * doc.duration
            reaction_time = np.random.uniform(0.02, 0.05)
            next_time = watch_time + reaction_time
        else:
            response.clicked = False
            if np.random.random() >= self.config["leave_prob"]:
                next_time = max(
                    np.random.exponential(self._user_state.next_time_mean), 1
                )
            else:
                next_time = np.random.uniform(0.02, 0.05)

        response.next_time = next_time

        # retention
        cur_day = self._user_state.cur_day
        new_day = self._user_state.get_day(
            self._user_state.cum_next_time + response.next_time
        )
        if new_day - cur_day == 1:
            response.retention = 1.0
        else:
            response.retention = 0.0

    def update_state(self, slate_documents, responses):
        """Update user state"""
        for doc, response in zip(slate_documents, responses):
            self._user_state.total_impress_num += 1
            # update time
            cur_day = self._user_state.cur_day
            self._user_state.cum_next_time += response.next_time
            new_day = self._user_state.cur_day

            if cur_day != new_day:
                self._user_state.last_day = cur_day
            # update user interest
            target = doc.emb - self._user_state.interest
            update = -self._user_state.alpha * target

            if response.clicked:
                update *= -1.0
                self._user_state.click_dict[cur_day] += 1

            self._user_state.impressed_dict[cur_day] += 1
            self._user_state.interest += update
            self._user_state.interest = np.clip(
                self._user_state.interest, 0.0, 1.0
            )

            # update user_next_time_lambda according to the last few days' watch num

            if (
                self._user_state.last_day is not None
                and self._user_state.cur_day != self._user_state.last_day
                and new_day != cur_day
            ):
                hist_watch_dict = {1: 0, 2: 0, 3: 0}
                for k in hist_watch_dict:
                    if new_day > k:
                        hist_watch_dict[k] = self._user_state.click_dict[
                            new_day - k
                        ]

                self._user_state.next_time_mean = (
                    self.config["next_time_mean"]
                    * 1.0
                    / max(
                        min(
                            (
                                (
                                    5.0 * hist_watch_dict[1]
                                    + 2.0 * hist_watch_dict[2]
                                    + hist_watch_dict[3]
                                )
                                - 3 * self.config["standard_click_num"]
                            )
                            / self.config["standard_click_num"],
                            20,
                        ),
                        0.2,
                    )
                )
                # print(
                #     (
                #         5.0 * hist_watch_dict[1]
                #         + 2.0 * hist_watch_dict[2]
                #         + hist_watch_dict[3]
                #     )
                #     / self.config["standard_click_num"]
                # )
                # print(
                #     "next_mean",
                #     self._user_state.next_time_mean,
                #     hist_watch_dict[1],
                # )
                # print("hist", hist_watch_dict)
                # print("click", self._user_state.click_dict)

    def is_terminal(self):
        """Returns a boolean indicating if the session is over."""
        return (
            self._user_state.cum_next_time >= self.config["time_budget"]
            or self._user_state.total_impress_num
            >= self.config["max_impress_num"]
        )


def create_env():
    user_model = UserModel(env_config)
    video_sampler = VideoSampler()
    # single user environment
    env = environment.Environment(
        user_model,
        video_sampler,
        env_config["candidate_size"],
        env_config["slate_size"],
        resample_documents=True,
    )

    return env
