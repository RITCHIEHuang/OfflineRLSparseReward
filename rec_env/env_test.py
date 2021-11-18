import matplotlib.pyplot as plt

from rec_env.env import VideoSampler, UserSampler


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


if __name__ == "__main__":
    # test_document()
    test_user()
