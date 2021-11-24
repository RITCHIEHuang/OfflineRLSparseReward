import numpy as np
from gym import spaces

from recsim import document
from recsim import user

from recsim.simulator import environment, recsim_gym

env_config = {
    # global
    "seed": 9999,
    "slate_size": 1,
    "candidate_size": 10,
    # video
    "video_num": 10000,
    "max_video_duration": 5.0,
    "min_video_duration": 1.0,
    "emb_dim": 10,
    # user
    "time_budget": 9 * 24 * 60,
    "next_time_mean": 600.0,
    "alpha_min": 0.001,
    "alpha_max": 0.01,
    "max_hist_len": 50,
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
        next_time_lambda,
        alpha=0.001,
        click_list=None,
        impressed_list=None,
    ):

        self.interest = interest
        self.next_time_lambda = next_time_lambda

        self.alpha = alpha
        self.click_list = click_list
        self.impressed_list = impressed_list

        self.cum_next_time = 0.0
        self.day = self.cum_next_time // 1440

    def create_observation(self):
        return {
            "interest": self.interest,
            # "cum_next_time": self.cum_next_time,
            # "day": self.day,
        }

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
        score = (self.interest @ doc_obs["emb"]) / np.sqrt(
            self.interest @ self.interest * doc_obs["emb"] @ doc_obs["emb"]
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
        state_parameters = {
            "alpha": self._rng.uniform(
                self.config["alpha_min"],
                self.config["alpha_max"],
            ),
            "click_list": [],
            "impressed_list": [],
            "interest": self._rng.uniform(
                0,
                1,
                env_config["feature_dim"],
            ),
            "next_time_lambda": self.config["next_time_mean"],
        }
        return self._user_ctor(**state_parameters)


class UserResponse(user.AbstractResponse):
    def __init__(self, clicked=False, next_time=0.0, retention=0.0):
        self.clicked = clicked
        self.next_time = next_time
        self.retention = retention

    def create_observation(self):
        return {
            "click": int(self.clicked),
            "next_time": np.array(self.next_time),
            "retention": np.array(self.retention),
        }

    @classmethod
    def response_space(cls):
        return spaces.Dict(
            {
                "click": spaces.Discrete(2),
                "next_time": spaces.Box(
                    low=0.0,
                    high=1440.0,
                    shape=tuple(),
                    dtype=np.float32,
                ),
                "retention": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=tuple(),
                    dtype=np.float32,
                ),
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
        # print(score)
        if score >= 0.7:
            # click case
            response.clicked = True
            watch_time = score * doc.duration
            reaction_time = np.random.uniform(0.02, 0.05)
            next_time = watch_time + reaction_time
        else:
            response.clicked = False
            next_time = np.random.exponential(
                self._user_state.next_time_lambda
            )

        response.next_time = next_time

        # reward
        cur_day = self._user_state.cum_next_time // 1440
        new_day = (self._user_state.cum_next_time + response.next_time) // 1440
        if new_day - cur_day == 1:
            response.retention = 1.0
        else:
            response.retention = 0.0

    def update_state(self, slate_documents, responses):
        """Update user state"""
        for doc, response in zip(slate_documents, responses):
            self._user_state.impressed_list.append(doc)

            # update user interest
            target = doc.emb - self._user_state.interest
            update = -self._user_state.alpha * target

            if response.clicked:
                update *= -1.0
                self._user_state.click_list.append(doc)

                self._user_state.click_list = self._user_state.click_list[
                    -self.config["max_hist_len"] :
                ]

            self._user_state.cum_next_time += response.next_time
            self._user_state.interest += update
            self._user_state.interest = np.clip(
                self._user_state.interest, 0.0, 1.0
            )

        self._user_state.impress_list = self._user_state.impressed_list[
            -self.config["max_hist_len"] :
        ]

    def is_terminal(self):
        """Returns a boolean indicating if the session is over."""
        return self._user_state.cum_next_time >= self.config["time_budget"]


def reward_fn(responses):
    total_reward = 0.0
    for response in responses:
        total_reward += response.retention
    return total_reward


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

    return recsim_gym.RecSimGymEnv(
        env,
        reward_fn,
    )


def get_flatten_obs(obs, env):
    return np.concatenate(
        [
            spaces.flatten(env.observation_space.spaces["user"], obs["user"]),
            spaces.flatten(env.observation_space.spaces["doc"], obs["doc"]),
        ]
    )


def make_recs_env(**kwargs):
    pass
