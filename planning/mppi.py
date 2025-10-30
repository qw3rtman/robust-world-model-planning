import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from einops import rearrange, repeat
from .base_planner import BasePlanner
from utils import move_to_device

def _ensure_non_zero(cost, beta, factor):
    return torch.exp(-factor * (cost - beta))

class MPPIPlanner(BasePlanner):
    """
    Model Predictive Path Integral control

    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning',
    based off of https://github.com/ferreirafabio/mppi_pendulum
    """
    def __init__(
        self,
        horizon,
        num_samples,
        var_scale,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        rollout_type,
        logging_prefix="plan_0",
        log_filename="logs.json",
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        self.rollout_type = rollout_type

        self.K = num_samples  # N_SAMPLES
        self.T = horizon  # TIMESTEPS

        self.lambda_ = 1.

        noise_mu = torch.zeros(self.action_dim)
        noise_sigma = var_scale * torch.eye(self.action_dim, self.action_dim)

        u_init = torch.zeros_like(noise_mu)

        self.noise_mu = noise_mu.to(self.device)
        self.noise_sigma = noise_sigma.to(self.device)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)
        self.u_init = u_init.to(self.device)

    def shift_nominal_trajectory(self, actions):
        """
        Shift the nominal trajectory forward one step
        """
        # shift command 1 time step
        actions = torch.roll(actions, -1, dims=0)
        actions[-1] = self.u_init
        return actions

    def command(self, obs_0, obs_g, actions, step):
        return self._command(obs_0, obs_g, actions, step)

    def _compute_weighting(self, cost_total):
        beta = torch.min(cost_total)
        cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)
        eta = torch.sum(cost_total_non_zero)
        omega = (1. / eta) * cost_total_non_zero
        return omega

    def _command(self, obs_0, obs_g, actions, step):
        cost_total, noise = self._compute_total_cost_batch(obs_0, obs_g, actions, step)

        omega = self._compute_weighting(cost_total)
        perturbations = torch.sum(omega.view(-1, 1, 1) * noise, dim=0)

        actions = actions + perturbations
        return actions

    def _compute_rollout_costs(self, obs_0, obs_g, perturbed_actions, step):
        num_samples, T, action_dim = perturbed_actions.shape
        assert action_dim == self.action_dim

        cost_total = torch.zeros(self.num_samples, device=self.device)

        obs_pred = self.wm.custom_rollout(obs_0=obs_0, actions=perturbed_actions.unsqueeze(2))
        c = self.objective_fn(obs_pred, obs_g, step=step)
        cost_total += c

        # action perturbation cost
        cost_total = cost_total
        return cost_total

    def _compute_perturbed_action_and_noise(self, actions):
        noise = self.noise_dist.rsample((self.K, self.T))
        perturbed_actions = actions + noise
        return perturbed_actions, noise

    def _compute_total_cost_batch(self, obs_0, obs_g, actions, step):
        perturbed_actions, noise = self._compute_perturbed_action_and_noise(actions)
        action_cost = self.lambda_ * noise @ self.noise_sigma_inv  # Like original paper

        rollout_cost = self._compute_rollout_costs(obs_0, obs_g, perturbed_actions, step)

        # action perturbation cost
        perturbation_cost = torch.sum(actions * action_cost, dim=(1, 2))
        cost_total = rollout_cost + perturbation_cost
        return cost_total, noise

    def plan(self, obs_0, obs_g, actions=None, step=0):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        
        z_obs_g = self.wm.encode_obs(trans_obs_g)
        n_evals = trans_obs_0["visual"].shape[0]
        # First iteration
        if actions == None:
            actions = self.noise_dist.sample((n_evals, self.T)) # [B, T, action_dim]
        else:
            # Append a zero action at the end
            actions = torch.cat((actions, self.noise_dist.sample((n_evals, 1))), dim=1)
        # trans_obs_0, z_obs_g: [B, state_dim]
        for traj in range(n_evals):
            cur_trans_obs_0 = {
                key: repeat(
                    arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                )
                for key, arr in trans_obs_0.items()
            }
            cur_z_obs_g = {
                key: repeat(
                    arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                )
                for key, arr in z_obs_g.items()
            }
            with torch.no_grad():
                actions[traj] = self.command(cur_trans_obs_0, cur_z_obs_g, actions[traj], step)

        return actions, np.full(n_evals, np.inf)  # all actions are valid