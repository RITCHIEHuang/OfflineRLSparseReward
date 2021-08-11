# pylint: disable=protected-access
import torch
from torch import nn


class RewardDecomposer(nn.Module):
    def __init__(self, obs_size, act_size, hidden_size=256, aux_loss_coef=0.5):
        super().__init__()

        self._aux_loss_coef = aux_loss_coef
        self.lstm = nn.LSTM(obs_size + act_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, 1)

    def compute_decomposed_reward(self, observations, actions):
        lstm_out, *_ = self.lstm(torch.cat([observations, actions], dim=-1))
        net_out = self.fc_out(lstm_out)

        return net_out

    def forward(self, observations, actions):
        return self.compute_decomposed_reward(observations, actions)

    def compute_error(
        self,
        pred_rews: torch.Tensor,
        rews: torch.Tensor,
    ) -> torch.Tensor:

        returns = rews.sum(dim=1)
        # Main task: predicting return at last timestep
        main_loss = torch.mean(pred_rews[:, -1] - returns) ** 2
        # Auxiliary task: predicting final return at every timestep ([..., None] is for correct broadcasting)
        aux_loss = torch.mean(pred_rews[:, :] - returns[..., None]) ** 2
        # Combine losses
        loss = main_loss + aux_loss * self._aux_loss_coef
        return loss.view(-1, 1)

    def compute_redistribued_reward(self, observations, actions, rewards):
        pred_rewards = self.compute_decomposed_reward(observations, actions)[
            ..., 0
        ]
        redistributed_reward = pred_rewards[:, 1:] - pred_rewards[:, :-1]
        # For the first timestep we will take (0-predictions[:, :1]) as redistributed reward
        redistributed_reward = torch.cat(
            [pred_rewards[:, :1], redistributed_reward], dim=1
        )

        # Calculate prediction error
        returns = rewards.sum(dim=1)
        predicted_returns = redistributed_reward.sum(dim=1)
        prediction_error = returns - predicted_returns

        # Distribute correction for prediction error equally over all sequence positions
        redistributed_reward += (
            prediction_error[:, None] / redistributed_reward.shape[1]
        )
        return redistributed_reward


if __name__ == "__main__":
    model = RewardDecomposer(10, 5)
    obs = torch.rand((2, 5, 10))
    act = torch.rand((2, 5, 5))
    pred_rew = model(obs, act)
    print(pred_rew.shape)
