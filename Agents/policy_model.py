#agents/policy_model.py

"""
defines a simple neural network policy to predict variable values in
equation . Stochastic policy model for REINFORCE:
Predicts mean and log_std to define a Normal distribution over solutions (x).

"""


# import all necessary files, libraries and modules
import torch
import torch.nn as nn
from torch.distributions import Normal


# this is gaussian REINFORCE policy
class GaussianPolicy(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=256, action_scale=25.0):  # Increased hidden dim
        super().__init__()
        self.action_scale = action_scale

        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # Add batch normalization
            # nn.Dropout(p=0.2),  # Reduced dropout

            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2)

        )
        self.mean_head = nn.Linear(hidden_dim // 2, 1)
        self.log_std = nn.Parameter(torch.zeros(1))  # learnable std (log)

    def forward(self, x: torch.Tensor, sample: bool = True):
        """
          :param x: [batch_size, input_dim] feature tensor
          :param sample: if True, sample from distribution, else return mean
          :return: action, log_prob
          """
        h = self.feature_net(x)
        mu = self.mean_head(h).squeeze(-1)

        std = torch.exp(torch.clamp(self.log_std, -4, 1))
        dist = Normal(mu, std)

        action = dist.rsample() if sample else mu
        action = torch.tanh(action) * self.action_scale
        log_prob = dist.log_prob(action / self.action_scale)

        return action.unsqueeze(-1), log_prob.unsqueeze(-1), dist.entropy().unsqueeze(-1)


def build_policy_model(lr=1e-4):
    """Factory function to create the model and its optimizer."""
    model = GaussianPolicy(input_dim=7)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            # nn.ReLU(),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            # nn.ReLU(),
            nn.LeakyReLU(0.01)
        )
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.layer_norm(x + self.block(x))


# actor-critic + GAE policy
class ActorCritic(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=256, action_scale=12.0):
        super().__init__()
        self.action_scale = action_scale

        # shared feature network
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.ReLU(),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim //2),
            # nn.ReLU(),
            nn.LeakyReLU(0.01),
            # nn.LayerNorm(hidden_dim),
        )
        self.actor_mean = nn.Linear(hidden_dim//2, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))

        #critic head
        self.critic = nn.Linear(hidden_dim//2, 1)

    def forward(self, x, sample=True):
        features = self.shared(x)

        #Actor output
        mu = self.actor_mean(features).squeeze(-1)
        std = torch.exp(self.actor_log_std)
        dist = Normal(mu, std)

        if sample:
            action = dist.rsample()
        else:
            action = mu

        # ✅ Use sigmoid instead of tanh
        scaled_action = torch.sigmoid(action) * self.action_scale

        # ✅ Remove tanh correction in log_prob
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)

        # Critic
        value = self.critic(features).squeeze(-1)

        return scaled_action.unsqueeze(-1), log_prob, value, entropy


def build_actor_critic(lr=1e-4):
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer
