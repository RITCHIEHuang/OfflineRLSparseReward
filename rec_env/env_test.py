import matplotlib.pyplot as plt

from rec_env.env import get_recs_env
from rec_env.recs_env import VideoSampler, UserSampler
from rec_env.gen_dataset import create_agent


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


def test_env():
    env = get_recs_env(
        obs_encode_format="vec",
    )

    print("=" * 88)
    print("env_obs_space", env.observation_space)
    print("env_act_space", env.action_space)

    agent = create_agent(env)
    obs = env.reset()
    print(obs.shape)

    print("=" * 88)
    print("init obs", obs)
    print("=" * 88)
    action = agent.begin_episode(obs)

    for _ in range(1):
        obs, reward, done, info = env.step(action)
        action = agent.step(reward, obs)

        print("obs", obs)
        print("rew", reward)
        print("done", done)
        print("info", info)

        print("=" * 88)


if __name__ == "__main__":
    # test_document()
    # test_user()
    test_env()
