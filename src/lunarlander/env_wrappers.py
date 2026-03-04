"""Gym wrappers for replacing the environment reward with a learned reward model."""

import numpy as np
import torch
import gymnasium as gym

from lunarlander.reward_model import RewardModel


class LearnedRewardWrapper(gym.Wrapper):
    """
    Replaces the environment's step reward with a score from a learned RewardModel.

    The reward model expects a feature vector extracted from (obs, action).
    Features: concatenation of flattened obs and one-hot action (for discrete envs).

    Args:
        env: the base gymnasium environment
        reward_model: trained RewardModel instance
        device: torch device for inference
        scale: multiply learned reward by this factor (default 1.0)
    """

    def __init__(
        self,
        env: gym.Env,
        reward_model: RewardModel,
        device: str = "cpu",
        scale: float = 1.0,
    ):
        super().__init__(env)
        self.reward_model = reward_model
        self.reward_model.eval()
        self.device = device
        self.scale = scale

        obs_dim = int(np.prod(env.observation_space.shape))
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.n_actions = env.action_space.n
            self.action_type = "discrete"
        else:
            self.n_actions = int(np.prod(env.action_space.shape))
            self.action_type = "continuous"

        self.feat_dim = obs_dim + self.n_actions

    def _make_features(self, obs: np.ndarray, action) -> torch.Tensor:
        obs_flat = obs.flatten().astype(np.float32)
        if self.action_type == "discrete":
            action_feat = np.zeros(self.n_actions, dtype=np.float32)
            action_feat[int(action)] = 1.0
        else:
            action_feat = np.array(action, dtype=np.float32).flatten()

        feat = np.concatenate([obs_flat, action_feat])
        return torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        info["env_reward"] = float(env_reward)

        feat = self._make_features(obs, action)
        with torch.no_grad():
            learned_reward = self.reward_model(feat).item() * self.scale

        return obs, learned_reward, terminated, truncated, info
