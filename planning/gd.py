import torch
import numpy as np
from einops import rearrange
from .base_planner import BasePlanner
from utils import move_to_device


class GDPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        action_noise,
        sample_type,
        lr,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        optimizer_type,
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
        self.action_noise = action_noise
        self.sample_type = sample_type
        self.lr = lr
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        self.optimizer_type = optimizer_type
        self.rollout_type = rollout_type

    def init_actions(self, trans_obs_0, trans_obs_g, actions=None):
        """
        Initializes or appends actions for planning, ensuring the output shape is (b, self.horizon, action_dim).
        """
        n_evals = trans_obs_0["visual"].shape[0]
        if actions is None:
            actions = torch.zeros(n_evals, 0, self.action_dim)
        device = actions.device
        t = actions.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            if self.sample_type == "randn":
                torch.cuda.manual_seed(617)
                torch.manual_seed(617)
                new_actions = torch.randn(n_evals, remaining_t, self.action_dim)
            elif self.sample_type == "zero":  # zero action of env
                new_actions = torch.zeros(n_evals, remaining_t, self.action_dim)
                new_actions = rearrange(
                    new_actions, "... (f d) -> ... f d", f=self.evaluator.frameskip
                )
                new_actions = self.preprocessor.normalize_actions(new_actions)
                new_actions = rearrange(new_actions, "... f d -> ... (f d)")
            elif self.sample_type == "initnet":
                self.wm.eval()
                with torch.no_grad():
                    z_goal = self.wm.encode_z(trans_obs_g)[:, 0, :, :]
                    z_0 = self.wm.encode_z(trans_obs_0)[:, 0, :, :]
                    z_0 = z_0.detach()
                    z_goal = z_goal.detach()
                    new_actions = self.wm.initnet(z_0, z_goal)
                    new_actions = new_actions[:, -remaining_t:, :]
            actions = torch.cat([actions, new_actions.to(device)], dim=1)
        return actions

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
            actions: (B, T, action_dim) torch.Tensor
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs(trans_obs_g)
        z_obs_g_detached = {key: value.detach() for key, value in z_obs_g.items()}

        actions = self.init_actions(trans_obs_0, trans_obs_g, actions).to(self.device)
        actions_init = actions.clone()

        actions.requires_grad = True
        optimizer = self.get_action_optimizer(actions, self.optimizer_type)
        n_evals = actions.shape[0]

        for i in range(self.opt_steps):
            optimizer.zero_grad()

            if self.rollout_type == "regular":
                i_z_obses, i_zs = self.wm.rollout_legacy(
                    obs_0=trans_obs_0,
                    act=actions,
                )
            elif self.rollout_type == "custom":
                i_z_obses = self.wm.custom_rollout(
                    obs_0=trans_obs_0,
                    actions=actions.unsqueeze(2),
                )

            loss = self.objective_fn(i_z_obses, z_obs_g_detached, step)
            total_loss = loss.mean() * n_evals  # loss for each eval is independent
            total_loss.backward()
            with torch.no_grad():
                actions_grad_norm = actions.grad.norm()
            optimizer.step()
            with torch.no_grad():
                actions += torch.randn_like(actions) * self.action_noise  # Add Gaussian noise

            self.wandb_run.log({
                f"{self.logging_prefix}/loss": total_loss.item(),
                f"{self.logging_prefix}/action_norm": actions.norm().item(),
                f"{self.logging_prefix}/actions_grad_norm": actions_grad_norm.item(),
                "step": i + 1,
            })

            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    actions.detach(), filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success
        return actions, np.full(n_evals, np.inf)  # all actions are valid
