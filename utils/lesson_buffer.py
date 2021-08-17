# -*- coding: utf-8 -*-
import numpy as np


class LessonBuffer:
    def __init__(self, size, max_time, n_features):
        self.size = size
        # Samples, time, features
        self.states_buffer = np.empty(shape=(size, max_time + 1, n_features))
        self.actions_buffer = np.empty(shape=(size, max_time))
        self.rewards_buffer = np.empty(shape=(size, max_time))
        self.lens_buffer = np.empty(shape=(size, 1), dtype=np.int32)
        self.next_spot_to_add = 0
        self.buffer_is_full = False
        self.samples_since_last_training = 0

    # LSTM training does only make sense, if there are sequences in the buffer which have different returns.
    # LSTM could otherwise learn to ignore the input and just use the bias units.
    def different_returns_encountered(self):
        if self.buffer_is_full:
            return np.unique(self.rewards_buffer[..., -1]).shape[0] > 1
        else:
            return (
                np.unique(
                    self.rewards_buffer[: self.next_spot_to_add, -1]
                ).shape[0]
                > 1
            )

    # We only train if 64 samples are played by a random policy
    def full_enough(self):
        return self.buffer_is_full or self.next_spot_to_add > 256

    # Add a new episode to the buffer
    def add(self, states, actions, rewards):
        traj_length = states.shape[0]
        next_ind = self.next_spot_to_add
        self.next_spot_to_add = self.next_spot_to_add + 1
        if self.next_spot_to_add >= self.size:
            self.buffer_is_full = True
        self.next_spot_to_add = self.next_spot_to_add % self.size
        self.states_buffer[next_ind, :traj_length] = states.squeeze()
        self.states_buffer[next_ind, traj_length:] = 0
        self.actions_buffer[next_ind, : traj_length - 1] = actions
        self.actions_buffer[next_ind, traj_length:] = 0
        self.rewards_buffer[next_ind, : traj_length - 1] = rewards
        self.rewards_buffer[next_ind, traj_length:] = 0
        self.lens_buffer[next_ind] = traj_length

    # Choose <batch_size> samples uniformly at random and return them.
    def sample(self, batch_size):
        self.samples_since_last_training = 0
        if self.buffer_is_full:
            indices = np.random.randint(0, self.size, batch_size)
        else:
            indices = np.random.randint(0, self.next_spot_to_add, batch_size)
        return (
            self.states_buffer[indices, :, :],
            self.actions_buffer[indices, :],
            self.rewards_buffer[indices, :],
            self.lens_buffer[indices, :],
        )
