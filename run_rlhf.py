import pathlib
from typing import Tuple

try:
    import gymnasium as gym
except ImportError:
    import gym
import numpy as np
import stable_baselines3 as sb3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from tqdm import trange

from data import load_data
from util import export_plot, np2torch


class RewardModel(nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dim: int, r_min: float, r_max: float
    ):
        """Initialize a reward model

        Parameters
        ----------
        obs_dim : int
            Dimension of the observation space
        action_dim : int
            Dimension of the action space
        hidden_dim : int
            Number of neurons in the hidden layer
        r_min : float
            Minimum reward value
        r_max : float
            Maximum reward value

        Defines self.net to be a neural network with a single hidden layer of size
        hidden_dim that takes as input an observation and an action and outputs a
        reward value. Uses LeakyRelu as hidden activation function. The output of the network
        is guaranteed to be in the interval [0, 1].
        """

        super().__init__()
        total_in_dim = obs_dim + action_dim

        self.net = nn.Sequential(
            nn.Linear(total_in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
            )
        
        self.optimizer = torch.optim.AdamW(self.net.parameters())
        self.r_min = r_min
        self.r_max = r_max

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward callback for the RewardModel

        Parameters
        ----------
        obs : torch.Tensor
            Batch of observations
        action : torch.Tensor
            Batch of actions

        Returns
        -------
        torch.Tensor
            Batch of predicted rewards

        Uses self.net to predict the rewards associated with the input
        (observation, action) pairs, and scales them so that they are
        within [self.r_min, self.r_max].

        """
        if obs.ndim == 3:
            B, T = obs.shape[:2]
            assert action.ndim == 3 and action.shape[:2] == (B, T)
            obs = obs.reshape(-1, obs.shape[-1])
            action = action.reshape(-1, action.shape[-1])
            needs_reshape = True
        else:
            needs_reshape = False

        rewards = torch.zeros(obs.shape[0])

        input_tensor = torch.cat((obs, action), -1)
        rewards = self.net(input_tensor)
        t_val = (self.r_max - self.r_min)
        rewards = self.r_min + (t_val * rewards)

        if needs_reshape:
            rewards = rewards.reshape(B, T)
        return rewards

    def compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Given an (observation, action) pair, return the predicted reward.

        Parameters
        ----------
        obs : np.ndarray (obs_dim, )
            A numpy array with an observation.
        action : np.ndarray (act_dim, )
            A numpy array with an action

        Returns
        -------
        float
            The predicted reward for the state-action pair.

        Returns the predicted reward for the given (observation, action) pair.
        """
        t_obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32), 0)
        t_action = torch.unsqueeze(torch.tensor(action, dtype=torch.float32), 0)
        reward = torch.squeeze(self.forward(t_obs, t_action))
        return reward.detach().numpy()


    def update(self, batch: Tuple[torch.Tensor]):
        """Given a batch of data, update the reward model.

        Parameters
        ----------
        batch : Tuple[torch.Tensor]
            A batch with two trajectories (observations and actions) and a label
            encoding which one is prefered (0 if it is the first one, 1 otherwise).

        Computes the cumulative predicted rewards for each trajectory, and calculates
        the loss following the Bradley-Terry preference model.
        """
        obs1, obs2, act1, act2, label = batch

        loss = torch.zeros(1)

        reward1 = torch.sum(self.forward(obs1, act1), dim=-1, keepdim=True)
        reward2 = torch.sum(self.forward(obs2, act2),  dim=-1, keepdim=True)
        c = torch.cat((reward1, reward2), -1)

        u1 = torch.unsqueeze(1 - label, 1)
        u2 = torch.unsqueeze(label, 1)
        u = torch.cat((u1, u2), -1)
        loss = F.cross_entropy(c, u)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class CustomRewardEnv(gym.Wrapper):
    def __init__(self, env, reward_fn):
        super().__init__(env)
        self.reward_fn = reward_fn
        self._obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        self._obs = obs.copy()
        return obs, info

    def step(self, action):
        next_obs, og_reward, terminated, truncated, info = self.env.step(action)
        reward = self.reward_fn.compute_reward(self._obs, action)
        info["og_reward"] = og_reward
        self._obs = next_obs.copy()
        return next_obs, reward, terminated, truncated, info


def evaluate(env, policy):
    model_return, og_return = 0, 0
    T = env.spec.max_episode_steps
    obs, _ = env.reset()
    for _ in range(T):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        model_return += reward
        og_return += info["og_reward"]
        if done:
            break
    return model_return, og_return


class EvalCallback(sb3.common.callbacks.BaseCallback):
    def __init__(self, eval_period, num_episodes, env, policy):
        super().__init__()
        self.eval_period = eval_period
        self.num_episodes = num_episodes
        self.env = env
        self.policy = policy

        self.original_returns = []
        self.learned_returns = []

    def _on_step(self):
        if self.n_calls % self.eval_period == 0:
            print(f"Evaluating after {self.n_calls} steps")
            model_returns, og_returns = [], []
            for _ in range(self.num_episodes):
                model_return, og_return = evaluate(self.env, self.policy)
                model_returns.append(model_return)
                og_returns.append(og_return)
            print(
                f"Model return: {np.mean(model_returns):.2f} +/- {np.std(model_returns):.2f}"
            )
            print(
                f"Original return: {np.mean(og_returns):.2f} +/- {np.std(og_returns):.2f}"
            )
            self.original_returns.append(np.mean(og_returns))
            self.learned_returns.append(np.mean(model_returns))

        # If the callback returns False, training is aborted early.
        return True


def main(args):
    output_path = pathlib.Path(__file__).parent.joinpath(
        "results_rlhf",
        f"Hopper-v4-rlhf-seed={args.seed}",
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_output = output_path.joinpath("model.zip")
    original_scores_output = output_path.joinpath("original_scores.npy")
    learned_scores_output = output_path.joinpath("learned_scores.npy")
    original_plot_output = output_path.joinpath("original_scores.png")
    learned_plot_output = output_path.joinpath("learned_scores.png")

    env = gym.make(args.env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    data = load_data(args.dataset_path)
    reward_model = RewardModel(
        obs_dim,
        action_dim,
        args.reward_model_hidden_dim,
        args.reward_min,
        args.reward_max,
    )

    print("Training reward model")
    for _ in trange(args.reward_model_steps):
        reward_model.update(data.sample(args.reward_model_batch_size))

    custom_reward_env = CustomRewardEnv(env, reward_model)
    agent = sb3.PPO("MlpPolicy", custom_reward_env, verbose=1)
    eval_callback = EvalCallback(
        args.eval_period,
        args.num_eval_episodes,
        custom_reward_env,
        lambda obs: agent.predict(obs)[0],
    )
    agent.learn(args.rl_steps, callback=eval_callback)

    # Log the results
    if not output_path.exists():
        output_path.mkdir(parents=True)
    agent.save(model_output)
    np.save(original_scores_output, eval_callback.original_returns)
    np.save(learned_scores_output, eval_callback.learned_returns)

    export_plot(
        eval_callback.original_returns,
        "Original returns",
        "Hopper-v4",
        original_plot_output,
    )
    export_plot(
        eval_callback.learned_returns,
        "Learned returns",
        "Hopper-v4",
        learned_plot_output,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--env-name", default="Hopper-v4")
    parser.add_argument(
        "--dataset-path",
        default=pathlib.Path(__file__).parent.joinpath("data", "prefs-hopper.npz"),
    )
    parser.add_argument("--reward-min", type=float, default=0.0)
    parser.add_argument("--reward-max", type=float, default=1.0)
    parser.add_argument("--reward-model-hidden-dim", type=int, default=64)
    parser.add_argument("--reward-model-steps", type=int, default=100000)
    parser.add_argument("--reward-model-batch-size", type=int, default=64)
    parser.add_argument("--rl-steps", type=int, default=1000000)
    parser.add_argument("--eval-period", type=int, default=1000)
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
