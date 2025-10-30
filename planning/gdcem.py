import torch
import numpy as np
from einops import rearrange, repeat
from .base_planner import BasePlanner
from utils import move_to_device


class GradCEMPlanner(BasePlanner):
    """
    GradCEM, implemented according to https://github.com/homangab/gradcem/tree/master
    """
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
        opt_steps,
        grad_steps,
        action_noise,
        lr,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        rollout_type,
        optimizer_type,
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
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.grad_steps = grad_steps
        self.optimizer_type = optimizer_type
        self.action_noise = action_noise
        self.lr = lr
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        self.rollout_type = rollout_type

    def init_mu_sigma(self, obs_0, actions=None):
        """
        actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        mu, sigma could depend on current obs, but obs_0 is only used for providing n_evals for now
        """
        n_evals = obs_0["visual"].shape[0]
        sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim])
        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions
        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
            mu = torch.cat([mu, new_mu.to(device)], dim=1)
        return mu, sigma

    def get_action_optimizer(self, actions, optimizer_type="sgd"):
        if optimizer_type == "sgd":
            return torch.optim.SGD([actions], lr=self.lr)
        elif optimizer_type == "momentum":
            return torch.optim.SGD([actions], lr=self.lr, momentum=0.9)
        elif optimizer_type == "adam":
            return torch.optim.Adam([actions], lr=self.lr)
        elif optimizer_type == "adamw":
            return torch.optim.AdamW([actions], lr=self.lr)
        else:
            raise ValueError(f"Invalid optimizer type: {optimizer_type}")

    def plan(self, obs_0, obs_g, actions=None, step=0):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs(trans_obs_g)

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]

        for i in range(self.opt_steps):
            # optimize individual instances
            losses = []
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
                cur_z_obs_g_detached = {key: value.detach() for key, value in cur_z_obs_g.items()}
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(
                        self.device
                    )
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]  # optional: make the first one mu itself
                actions = action.unsqueeze(2).detach().requires_grad_(True)
                # Create a new optimizer each time since we have a "new" sampled actions tensor
                optimizer = self.get_action_optimizer(actions, self.optimizer_type)
                for i in range(self.grad_steps):
                    optimizer.zero_grad()
                    if self.rollout_type == "regular":
                        i_z_obses, i_zs = self.wm.rollout_legacy(
                            obs_0=cur_trans_obs_0,
                            act=actions.squeeze(2),
                        )
                    elif self.rollout_type == "custom":
                        i_z_obses = self.wm.custom_rollout(
                            obs_0=cur_trans_obs_0,
                            actions=actions,
                        )
                    loss = self.objective_fn(i_z_obses, cur_z_obs_g_detached, step)
                    total_loss = loss.mean()
                    total_loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        actions += torch.randn_like(actions) * self.action_noise  # Add Gaussian noise
                with torch.no_grad():
                    if self.rollout_type == "regular":
                        i_z_obses, i_zs = self.wm.rollout_legacy(
                            obs_0=cur_trans_obs_0,
                            act=actions.squeeze(2),
                        )
                    elif self.rollout_type == "custom":
                        i_z_obses = self.wm.custom_rollout(
                            obs_0=cur_trans_obs_0,
                            actions=actions,
                        )
                    loss = self.objective_fn(i_z_obses, cur_z_obs_g_detached, step)
                    topk_idx = torch.argsort(loss)[: self.topk]
                    topk_action = action[topk_idx]
                    losses.append(loss[topk_idx[0]].item())
                    mu[traj] = topk_action.mean(dim=0)
                    sigma[traj] = topk_action.std(dim=0)

            del optimizer
            del actions

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": np.mean(losses), "step": i + 1}
            )
            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu, filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success

        return mu, np.full(n_evals, np.inf)  # all actions are valid