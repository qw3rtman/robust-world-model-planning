import os
import time
import hydra
import torch
import wandb
import logging
import warnings
import threading
import itertools
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from einops import rearrange
from accelerate import Accelerator
from torchvision import utils
import torch.distributed as dist
from pathlib import Path
from collections import OrderedDict
from hydra.types import RunMode
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from utils import slice_trajdict_with_t, cfg_to_dict, seed, move_to_device

# for Online World Modeling
from planning.gd import GDPlanner
from plan import DummyWandbRun
from preprocessor import Preprocessor
from datasets.img_transforms import default_transform
from planning.objectives import create_objective_fn
from collections import defaultdict

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        with open_dict(cfg):
            cfg["saved_folder"] = os.getcwd()
            log.info(f"Model saved dir: {cfg['saved_folder']}")
        cfg_dict = cfg_to_dict(cfg)
        model_name = cfg_dict["saved_folder"].split("outputs/")[-1]
        model_name += f"_{self.cfg.env.name}_f{self.cfg.frameskip}_h{self.cfg.num_hist}_p{self.cfg.num_pred}"

        if HydraConfig.get().mode == RunMode.MULTIRUN:
            log.info(" Multirun setup begin...")
            log.info(f"SLURM_JOB_NODELIST={os.environ['SLURM_JOB_NODELIST']}")
            log.info(f"DEBUGVAR={os.environ['DEBUGVAR']}")
            # ==== init ddp process group ====
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            try:
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    timeout=timedelta(minutes=5),  # Set a 5-minute timeout
                )
                log.info("Multirun setup completed.")
            except Exception as e:
                log.error(f"DDP setup failed: {e}")
                raise
            torch.distributed.barrier()
            # # ==== /init ddp process group ====

        self.accelerator = Accelerator(log_with="wandb")
        log.info(
            f"rank: {self.accelerator.local_process_index}  model_name: {model_name}"
        )
        self.device = self.accelerator.device
        log.info(f"device: {self.device}   model_name: {model_name}")
        self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.num_reconstruct_samples = self.cfg.training.num_reconstruct_samples
        self.total_epochs = self.cfg.training.epochs
        self.epoch = 0

        assert cfg.training.batch_size % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Batch_size: {cfg.training.batch_size} num_processes: {self.accelerator.num_processes}."
        )

        OmegaConf.set_struct(cfg, False)
        cfg.effective_batch_size = cfg.training.batch_size
        cfg.gpu_batch_size = cfg.training.batch_size // self.accelerator.num_processes
        OmegaConf.set_struct(cfg, True)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            wandb_run_id = None
            if os.path.exists("hydra.yaml"):
                existing_cfg = OmegaConf.load("hydra.yaml")
                wandb_run_id = existing_cfg["wandb_run_id"]
                log.info(f"Resuming Wandb run {wandb_run_id}")

            wandb_dict = OmegaConf.to_container(cfg, resolve=True)
            if self.cfg.debug:
                log.info("WARNING: Running in debug mode...")
                self.wandb_run = wandb.init(
                    project="dino_wm_debug",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            else:
                self.wandb_run = wandb.init(
                    project="dino_wm",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            OmegaConf.set_struct(cfg, False)
            cfg.wandb_run_id = self.wandb_run.id
            OmegaConf.set_struct(cfg, True)
            wandb.run.name = "{}".format(model_name)
            # Define separate step metrics for different phases
            # Step metrics
            wandb.define_metric("pretrain_initnet_step", summary="max")
            wandb.define_metric("train_step", summary="max")
            wandb.define_metric("val_step", summary="max")
            # Map metric groups to their respective step metrics
            wandb.define_metric("pretrain_initnet/*", step_metric="pretrain_initnet_step")
            wandb.define_metric("pretrain_initnet_*", step_metric="pretrain_initnet_step")
            wandb.define_metric("train_*", step_metric="train_step")
            wandb.define_metric("train/*", step_metric="train_step")
            wandb.define_metric("val_*", step_metric="val_step")
            wandb.define_metric("val/*", step_metric="val_step")
            with open(os.path.join(os.getcwd(), "hydra.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))

        seed(cfg.training.seed)
        # log.info(f"Loading dataset from {self.cfg.env.dataset.data_path} ...")
        self.datasets, traj_dsets = hydra.utils.call(
            self.cfg.env.dataset,
            num_hist=self.cfg.num_hist,
            num_pred=self.cfg.num_pred,
            N=self.cfg.N,
            frameskip=self.cfg.frameskip,
            pretrain_ratio=self.cfg.env.dataset.pretrain_ratio,
        )
        self.num_frames = self.cfg.num_hist + self.cfg.num_pred
        self.traj_len = self.cfg.N + self.num_frames - 1
        self.num_patches = 0
        self.train_traj_dset = traj_dsets["train"]
        self.val_traj_dset = traj_dsets["valid"]
        # self.pretrain_traj_dset = traj_dsets["pretrain"]
        self.radii = {}

        self.dataloaders = {}
        for x in ["train", "valid"]:#, "pretrain"]:
            if x == "pretrain":
                batch_size = self.cfg.training.pretrain_batch_size
            else:
                batch_size = self.cfg.training.batch_size
            
            self.dataloaders[x] = torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=batch_size,
                shuffle=False, # already shuffled in TrajSlicerDataset
                num_workers=self.cfg.env.num_workers,
                collate_fn=None,
            )

        log.info(f"dataloader batch size: {self.cfg.gpu_batch_size}")

        self.dataloaders["train"], self.dataloaders["valid"] = self.accelerator.prepare(
            self.dataloaders["train"], self.dataloaders["valid"]
        )

        self.encoder = None
        self.action_encoder = None
        self.proprio_encoder = None
        self.predictor = None
        self.decoder = None
        self.initnet = None
        self.train_encoder = self.cfg.model.train_encoder
        self.train_predictor = self.cfg.model.train_predictor
        self.train_decoder = self.cfg.model.train_decoder
        log.info(f"Train encoder, predictor, decoder:\
            {self.cfg.model.train_encoder}\
            {self.cfg.model.train_predictor}\
            {self.cfg.model.train_decoder}")

        self._keys_to_save = [
            "epoch",
        ]
        self._keys_to_save += (
            ["encoder", "encoder_optimizer"] if self.train_encoder else []
        )
        self._keys_to_save += (
            ["predictor", "predictor_optimizer"]
            if self.train_predictor and self.cfg.has_predictor
            else []
        )
        self._keys_to_save += (
            ["decoder", "decoder_optimizer"] if self.train_decoder else []
        )
        self._keys_to_save += ["action_encoder", "proprio_encoder"]
        self._keys_to_save += (
            ["initnet", "initnet_optimizer"] if self.cfg.model.train_initnet else []
        )

        self.init_models()
        self.init_optimizers()

        self.epoch_log = OrderedDict()
        self.pretrain_initnet_step = 0
        self.train_step = 0
        self.val_step = 0

    def save_ckpt(self, tag=None):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            ckpt = {}
            for k in self._keys_to_save:
                print(k)
                if hasattr(self.__dict__[k], "module"):
                    ckpt[k] = self.accelerator.unwrap_model(self.__dict__[k])
                else:
                    ckpt[k] = self.__dict__[k]
            torch.save(ckpt, "checkpoints/model_latest.pth")

            _tag = f"_{tag}" if tag is not None else ""
            model_name = f"checkpoints/model_{self.epoch}{_tag}.pth"
            torch.save(ckpt, model_name)
            log.info("Saved model to {}".format(os.getcwd()))
            ckpt_path = os.path.join(os.getcwd(), f"checkpoints/model_{self.epoch}.pth")
        else:
            ckpt_path = None
        model_name = self.cfg["saved_folder"].split("outputs/")[-1]
        model_epoch = self.epoch
        return ckpt_path, model_name, model_epoch

    def load_ckpt(self, filename="model_latest.pth"):
        ckpt = torch.load(filename, weights_only=False)
        for k, v in ckpt.items():
            """
            if k == "action_encoder":
                print(v.patch_embed.bias)
            """
            self.__dict__[k] = v
        not_in_ckpt = set(self._keys_to_save) - set(ckpt.keys())
        if len(not_in_ckpt):
            log.warning("Keys not found in ckpt: %s", not_in_ckpt)

    def init_models(self):
        model_ckpt = Path(to_absolute_path(self.cfg.ckpt_path)) / "checkpoints" / "model_latest.pth"
        if model_ckpt.exists():
            print("Loading ckpt from", model_ckpt)
            self.load_ckpt(model_ckpt)
            log.info(f"Resuming from epoch {self.epoch}: {model_ckpt}")

        # initialize encoder
        if self.encoder is None:
            self.encoder = hydra.utils.instantiate(
                self.cfg.encoder,
            )
        if not self.train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.proprio_encoder is None:
            self.proprio_encoder = hydra.utils.instantiate(
                self.cfg.proprio_encoder,
                in_chans=self.datasets["train"].proprio_dim,
                emb_dim=self.cfg.proprio_emb_dim,
            )
        self.proprio_dim = self.datasets["train"].proprio_dim
        self.proprio_emb_dim = self.cfg.proprio_emb_dim
        print(f"Proprio encoder type: {type(self.proprio_encoder)}")
        self.proprio_encoder = self.accelerator.prepare(self.proprio_encoder)

        if self.action_encoder is None:
            self.action_encoder = hydra.utils.instantiate(
                self.cfg.action_encoder,
                in_chans=self.datasets["train"].action_dim,
                emb_dim=self.cfg.action_emb_dim,
            )
        self.action_emb_dim = self.cfg.action_emb_dim
        self.action_encoder = self.accelerator.prepare(self.action_encoder)

        if self.accelerator.is_main_process:
            self.wandb_run.watch(self.action_encoder)
            self.wandb_run.watch(self.proprio_encoder)

        # initialize predictor
        if self.encoder.latent_ndim == 1:  # if feature is 1D
            self.num_patches = 1
        else:
            decoder_scale = 16  # from vqvae
            num_side_patches = self.cfg.img_size // decoder_scale
            self.num_patches = num_side_patches**2

        if self.cfg.concat_dim == 0:
            self.num_patches += 2

        if self.cfg.has_predictor:
            if self.predictor is None:
                self.predictor = hydra.utils.instantiate(
                    self.cfg.predictor,
                    num_patches=self.num_patches,
                    num_frames=self.cfg.num_hist,
                    dim=self.encoder.emb_dim
                    + (
                        self.proprio_emb_dim * self.cfg.num_proprio_repeat
                        + self.action_emb_dim * self.cfg.num_action_repeat
                    )
                    * (self.cfg.concat_dim),
                )
            if not self.train_predictor:
                for param in self.predictor.parameters():
                    param.requires_grad = False

        # initialize decoder
        if self.cfg.has_decoder:
            if self.decoder is None:
                if self.cfg.env.decoder_path is not None:
                    decoder_path = os.path.join(
                        self.base_path, self.cfg.env.decoder_path
                    )
                    ckpt = torch.load(decoder_path)
                    if isinstance(ckpt, dict):
                        self.decoder = ckpt["decoder"]
                    else:
                        self.decoder = torch.load(decoder_path)
                    log.info(f"Loaded decoder from {decoder_path}")
                else:
                    self.decoder = hydra.utils.instantiate(
                        self.cfg.decoder,
                        emb_dim=self.encoder.emb_dim,  # 384
                    )
            if not self.train_decoder:
                for param in self.decoder.parameters():
                    param.requires_grad = False

        if self.cfg.model.train_initnet:
            if self.initnet is None:
                self.initnet = hydra.utils.instantiate(
                    self.cfg.initnet,
                    in_channel=2*(self.encoder.emb_dim + self.cfg.proprio_emb_dim),
                    num_actions=self.traj_len,
                    action_dim=self.action_emb_dim,
                )
        self.encoder, self.predictor, self.decoder, self.initnet = self.accelerator.prepare(
            self.encoder, self.predictor, self.decoder, self.initnet
        )
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            decoder=self.decoder,
            initnet=self.initnet,
            proprio_dim=self.proprio_emb_dim,
            action_dim=self.action_emb_dim,
            concat_dim=self.cfg.concat_dim,
            num_action_repeat=self.cfg.num_action_repeat,
            num_proprio_repeat=self.cfg.num_proprio_repeat,
        )

    def init_optimizers(self):
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.cfg.training.encoder_lr,
        )
        self.encoder_optimizer = self.accelerator.prepare(self.encoder_optimizer)
        if self.cfg.has_predictor:
            self.predictor_optimizer = torch.optim.AdamW(
                self.predictor.parameters(),
                lr=self.cfg.training.predictor_lr,
            )
            self.predictor_optimizer = self.accelerator.prepare(
                self.predictor_optimizer
            )

            self.action_encoder_optimizer = torch.optim.AdamW(
                itertools.chain(
                    self.action_encoder.parameters(), self.proprio_encoder.parameters()
                ),
                lr=self.cfg.training.action_encoder_lr,
            )
            self.action_encoder_optimizer = self.accelerator.prepare(
                self.action_encoder_optimizer
            )

        if self.cfg.has_decoder:
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(), lr=self.cfg.training.decoder_lr
            )
            self.decoder_optimizer = self.accelerator.prepare(self.decoder_optimizer)
    
        if self.cfg.model.train_initnet:
            self.initnet_optimizer = torch.optim.Adam(
                self.initnet.parameters(),
                lr=self.cfg.training.initnet_lr,
            )
            self.initnet_optimizer = self.accelerator.prepare(self.initnet_optimizer)

    def monitor_jobs(self, lock):
        """
        check planning eval jobs' status and update logs
        """
        while True:
            with lock:
                finished_jobs = [
                    job_tuple for job_tuple in self.job_set if job_tuple[2].done()
                ]
                for epoch, job_name, job in finished_jobs:
                    result = job.result()
                    log_data = {
                        f"{job_name}/{key}": value for key, value in result.items()
                    }
                    log_data["epoch"] = epoch
                    self.wandb_run.log(log_data)
                    self.job_set.remove((epoch, job_name, job))
            time.sleep(1)

    def run(self):
        if self.accelerator.is_main_process:
            executor = ThreadPoolExecutor(max_workers=4)
            self.job_set = set()
            lock = threading.Lock()

            self.monitor_thread = threading.Thread(
                target=self.monitor_jobs, args=(lock,), daemon=True
            )
            self.monitor_thread.start()

        self.dagger_cache = defaultdict(list)

        self.epoch = 0
        self.save_ckpt()
        init_epoch = self.epoch + 1  # epoch starts from 1
        for epoch in range(init_epoch, init_epoch + self.total_epochs):
            self.epoch = epoch
            self.accelerator.wait_for_everyone()

            if self.cfg.model.train_initnet:
                self.pretrain_initnet()
                self.accelerator.wait_for_everyone()

            if self.cfg.method == "offline":
                self.train_wm()
            elif self.cfg.method == "online":
                if self.cfg.env.name == "pusht":
                    from datasets.pusht_dset import ACTION_MEAN, ACTION_STD, STATE_MEAN, STATE_STD, PROPRIO_MEAN, PROPRIO_STD
                    from env.pusht.pusht_wrapper import PushTWrapper
                    self.env = PushTWrapper()
                elif self.cfg.env.name == "wall":
                    from env.wall.wall_env_wrapper import WallEnvWrapper
                    from datasets.wall_dset import ACTION_MEAN, ACTION_STD, STATE_MEAN, STATE_STD
                    PROPRIO_MEAN = torch.zeros(self.datasets["train"].proprio_dim)
                    PROPRIO_STD = torch.ones(self.datasets["train"].proprio_dim)
                    self.env = WallEnvWrapper()
                elif self.cfg.env.name == "point_maze":
                    from env.pointmaze.point_maze_wrapper import PointMazeWrapper
                    ACTION_MEAN, ACTION_STD = self.datasets["train"].action_mean, self.datasets["train"].action_std
                    STATE_MEAN, STATE_STD = self.datasets["train"].state_mean, self.datasets["train"].state_std
                    PROPRIO_MEAN, PROPRIO_STD = self.datasets["train"].proprio_mean, self.datasets["train"].proprio_std
                    self.env = PointMazeWrapper()
                else:
                    raise Exception("Must setup environment manually here.")

                self.planner = GDPlanner(
                    wm=self.model,
                    action_dim=self.action_emb_dim,
                    objective_fn=create_objective_fn(alpha=1, base=2, mode='last'),
                    preprocessor=Preprocessor(
                        action_mean=ACTION_MEAN,
                        action_std=ACTION_STD,
                        state_mean=STATE_MEAN,
                        state_std=STATE_STD,
                        proprio_mean=PROPRIO_MEAN,
                        proprio_std=PROPRIO_STD,
                        transform=default_transform(), # we don't want to double transform the images
                    ),
                    evaluator=None, # NOTE: disable evaluation
                    wandb_run=DummyWandbRun(),

                    horizon=5,
                    action_noise=0.003,
                    sample_type='randn',
                    rollout_type="regular",
                    optimizer_type="adam",
                    lr=3e-1,
                    opt_steps=100,
                    eval_every=0,
                )
                self.train_online_wm()
            else:
                self.train_adversarial_wm()

            self.accelerator.wait_for_everyone()

            # self.val() # NOTE: uncomment to get validation loss
            ckpt_path, model_name, model_epoch = self.save_ckpt()

    def err_eval_single(self, z_pred, z_tgt):
        logs = {}
        for k in z_pred.keys():
            loss = self.model.emb_criterion(z_pred[k], z_tgt[k])
            logs[k] = loss
        return logs

    def err_eval(self, z_out, z_tgt, state_tgt=None):
        """
        z_pred: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        z_tgt: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        state:  (b, n_hist, dim)
        """
        logs = {}
        slices = {
            "full": (None, None),
            "pred": (-self.model.num_pred, None),
            "next1": (-self.model.num_pred, -self.model.num_pred + 1),
        }
        for name, (start_idx, end_idx) in slices.items():
            z_out_slice = slice_trajdict_with_t(
                z_out, start_idx=start_idx, end_idx=end_idx
            )
            z_tgt_slice = slice_trajdict_with_t(
                z_tgt, start_idx=start_idx, end_idx=end_idx
            )
            z_err = self.err_eval_single(z_out_slice, z_tgt_slice)

            logs.update({f"z_{k}_err_{name}": v for k, v in z_err.items()})

        return logs

    def pretrain_initnet(self):
        for i, data in enumerate(
            tqdm(self.dataloaders["pretrain"], desc=f"Epoch {self.epoch} Pretrain Initnet")
        ):
            self.pretrain_initnet_step += 1
            obs, act, state, goal, T = data
            batch_size = act.shape[0]
            self.model.train()
            
            act = act.unfold(1, self.num_frames, 1) # (B, N, action_dim, num_frames)
            act = act.permute(1, 0, 3, 2) # (N, B, num_frames, action_dim)
            
            act_tgt = torch.zeros((self.traj_len, batch_size, self.action_emb_dim)).to(self.device)
            
            obs_0 = {"visual": obs["visual"][:, 0, ...], "proprio": obs["proprio"][:, 0, ...]}
            z = self.model.encode(obs_0, act[0]) # (B, num_frames, num_patches, emb_dim + action_emb_dim)
            z_0 = z[:, 0, :, :-self.action_emb_dim] # (B, num_patches, emb_dim)
            act_tgt[:self.num_frames, :] = z.permute(1, 0, 2, 3)[:, :, 0, -self.action_emb_dim:] # (num_frames, B, action_emb_dim)
            for j in range(self.cfg.N - 1):
                act_tgt[j + self.num_frames] = self.model.encode_act(act[j + 1])[:, -1, :] # (B, num_frames, action_emb_dim)
            act_tgt = act_tgt.transpose(0, 1) # (B, N + num_frames, action_emb_dim)

            goal = {"visual": goal['visual'].unsqueeze(1), "proprio": goal['proprio'].unsqueeze(1)} # (B, 3, img_size, img_size), (B, proprio_dim)
            z_goal = self.model.encode_z(goal)[:, 0, :, :] # (B, num_patches, emb_dim)

            act_hat = self.initnet(z_0, z_goal) # (B, N + num_frames, action_emb_dim)

            loss = self.model.emb_criterion(act_hat, act_tgt)

            self.initnet_optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.initnet_optimizer.step()

            # Gather for logging only after stepping
            loss_log = self.accelerator.gather_for_metrics(loss.detach()).mean()
            err_log = {"pretrain_initnet/loss": [loss_log.item()]}
            self.logs_update(err_log, phase="pretrain_initnet")

    def train_online_wm(self):
        def write_to_cache(cache, i, obs, act):
            cache[i].append(({k: v.detach().cpu() for k, v in obs.items()}, act.detach().cpu()))

        def load_from_cache(cache, i):
            for obs, act in cache[i]:
                yield {k: v.to(self.device) for k, v in obs.items()}, act.to(self.device)

        for i, data in enumerate(
            tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train Online WM")
        ):
            self.train_step += 1
            original_visual, obs, act, state, goal, T = data
            plot = (i % 10 == 0) and self.accelerator.is_main_process
            self.model.train()

            if i % self.cfg.replay_buffer_frequency == 0: # and self.epoch % 10 == 1:
                init_states = state[:, 0, 0].detach().cpu().numpy() # NOTE: for simulator. B x 7 for PushT
                obs_0_for_simulator = {
                    'visual': (original_visual[:, :1, 0].permute(0, 1, 3, 4, 2).detach().cpu().numpy() * 255.0).astype(np.uint8),
                    'proprio': (obs['proprio'][:, :1, 0].detach().cpu().numpy() * self.planner.preprocessor.proprio_std.numpy() + self.planner.preprocessor.proprio_mean.numpy())
                }

                obs_g_for_simulator = {
                    'visual': (original_visual[:, -1:, 0].permute(0, 1, 3, 4, 2).detach().cpu().numpy() * 255.0).astype(np.uint8),
                    'proprio': (obs['proprio'][:, -1:, 0].detach().cpu().numpy() * self.planner.preprocessor.proprio_std.numpy() + self.planner.preprocessor.proprio_mean.numpy())
                }

                # 1. get act_hat by doing a GD plan from obs_0 to obs_g
                actions, actions_len = self.planner.plan(obs_0_for_simulator, obs_g_for_simulator, actions=None)
                exec_actions = self.planner.preprocessor.denormalize_actions(
                    rearrange(actions.detach().cpu(), "b t (f d) -> b (t f) d", f=5)
                ).numpy()

                # 2. rollout the model at each o_t with act_hat to get o_{t+1}
                rollout_obs = []
                for b in range(actions.shape[0]): # across the batch
                    _rollout_obs, _ = self.env.rollout(
                        self.cfg.training.seed,
                        init_states[b],
                        exec_actions[b]
                    )
                    rollout_obs.append(_rollout_obs)
                rollout_obs = {
                    'visual': np.concatenate([rollout['visual'][None, :] for rollout in rollout_obs], axis=0),
                    'proprio': np.concatenate([rollout['proprio'][None, :] for rollout in rollout_obs], axis=0)
                }

                # the planned actions
                _act = torch.hstack([act[:, :self.cfg['num_hist']], torch.tensor(actions, device=act.device)]) # use the ground truth action history
                _act = _act.unfold(1, self.num_frames, 1) # (B, N, action_dim, num_frames)
                _act = rearrange(_act, "B N action_dim num_frames -> (B N) num_frames action_dim")

                obs_for_wm = move_to_device(self.planner.preprocessor.transform_obs(rollout_obs), self.device)

                visual_obs = torch.hstack([
                    obs['visual'][:, 0, :self.cfg['num_hist']],
                    obs_for_wm['visual'][:, :-1:5]
                ])
                proprio_obs = torch.hstack([
                    obs['proprio'][:, 0, :self.cfg['num_hist']], 
                    obs_for_wm['proprio'][:, :-1:5]
                ])

                # the ground-truth/simulated observations
                _obs = {
                    "visual": rearrange(visual_obs.unfold(1, self.num_frames, 1).permute(0, 1, 5, 2, 3, 4), "b n f c h w -> (b n) f c h w"),
                    "proprio": rearrange(proprio_obs.unfold(1, self.num_frames, 1), "b n f proprio_dim -> (b n) f proprio_dim")
                }

                write_to_cache(self.dagger_cache, i // self.cfg.replay_buffer_frequency , _obs, _act)

            gt_act = act.unfold(1, self.num_frames, 1) # (B, N, action_dim, num_frames)
            gt_act = rearrange(gt_act, "B N action_dim num_frames -> (B N) num_frames action_dim")
            gt_obs = {"visual": rearrange(obs['visual'], "b n f c h w -> (b n) f c h w"), "proprio": rearrange(obs['proprio'], "b n f proprio_dim -> (b n) f proprio_dim")}

            iterator = itertools.chain(
                iter([(gt_obs, gt_act)]),
                load_from_cache(self.dagger_cache, i // self.cfg.replay_buffer_frequency)
            )

            # then do normal teacher forcing on the WM; i.e., WM(o_t, act_hat) -> o_{t+1}
            losss, loss_componentss = [], []
            pgd_losss, visual_epss, proprio_epss, action_epss = [], [], [], []
            for _j, (obs, act) in enumerate(iterator):
                if _j == 1:
                    torch.cuda.empty_cache()
                z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                    obs, act
                )

                self.encoder_optimizer.zero_grad()
                if self.cfg.has_decoder:
                    self.decoder_optimizer.zero_grad()
                if self.cfg.has_predictor:
                    self.predictor_optimizer.zero_grad()
                    self.action_encoder_optimizer.zero_grad()

                self.accelerator.backward(loss)

                if self.model.train_encoder:
                    self.encoder_optimizer.step()
                if self.cfg.has_decoder and self.model.train_decoder:
                    self.decoder_optimizer.step()
                if self.cfg.has_predictor and self.model.train_predictor:
                    self.predictor_optimizer.step()
                    self.action_encoder_optimizer.step()

                loss = self.accelerator.gather_for_metrics(loss).mean()
                loss_components = self.accelerator.gather_for_metrics(loss_components)
                loss_components = {
                    key: value.mean().item() for key, value in loss_components.items()
                }

                losss.append(loss.item())
                loss_componentss.append(loss_components)

            wm_losss = [loss_components["z_loss"] for loss_components in loss_componentss]
            decoder_loss = np.mean([loss_components["decoder_loss_reconstructed"] for loss_components in loss_componentss])
            loss = np.mean([loss for loss in losss])

            logs = {
                "train/wm_gt_loss": [wm_losss[0]],
                "train/decoder_loss": [decoder_loss],
                "train/loss": [loss],
            }

            if len(wm_losss) > 1:
                logs["train/wm_dagger_loss"] = [np.mean(wm_losss[1:])]

            self.logs_update(logs, phase="train")

            if self.cfg.has_decoder and plot:
                # obs["visual"]: (80,      4, 3, 224, 224)
                #                (B*(T-3), 4, 3, 224, 224)

                # visual_out:   (80,      3, 3, 224, 224)
                #               (B*(T-3), 3, 3, 224, 224)

                # visual_reconstructed: (80,      4, 3, 224, 224)
                #                       (B*(T-3), 4, 3, 224, 224)
                try:
                    self.plot_samples(
                        _obs["visual"],
                        visual_out,
                        visual_reconstructed,
                        self.epoch,
                        batch=i,
                        num_samples=self.num_reconstruct_samples,
                        phase="train",
                    )
                except:
                    pass

    def fgsm_step(self, obs, act, visual_eps_factor, proprio_eps_factor, action_eps_factor):
        # Gradient Ascent
        # ----------------------------------------------------------------------
        z = self.model.encode(obs, act)
        z_src = z[:, : self.model.num_hist, :, :]

        # radii_method
        if self.cfg.radii_method == "adaptive":
            visual_eps = z_src[..., :self.model.encoder.emb_dim].detach().reshape(z_src.shape[0], -1).std(dim=-1).mean() * visual_eps_factor
            proprio_eps = z_src[..., self.model.encoder.emb_dim:-self.model.action_dim].detach().reshape(z_src.shape[0], -1).std(dim=-1).mean() * proprio_eps_factor
            action_eps = z_src[..., -self.model.action_dim:].detach().reshape(z_src.shape[0], -1).std(dim=-1).mean() * action_eps_factor
        else: # fixed radii (first)
            if 'visual' not in self.radii:
                self.radii['visual'] = z_src[..., :self.model.encoder.emb_dim].detach().reshape(z_src.shape[0], -1).std(dim=-1).mean()
            if 'proprio' not in self.radii:
                self.radii['proprio'] = z_src[..., self.model.encoder.emb_dim:-self.model.action_dim].detach().reshape(z_src.shape[0], -1).std(dim=-1).mean()
            if 'action' not in self.radii:
                self.radii['action'] = z_src[..., -self.model.action_dim:].detach().reshape(z_src.shape[0], -1).std(dim=-1).mean()

            visual_eps = self.radii['visual'] * visual_eps_factor
            proprio_eps = self.radii['proprio'] * proprio_eps_factor
            action_eps = self.radii['action'] * action_eps_factor

        visual_alpha, propio_alpha, action_alpha = 1.25 * visual_eps, 1.25 * proprio_eps, 1.25 * action_eps

        delta = torch.empty(act.shape[0], self.model.num_hist, 196, self.model.emb_dim, device=self.device)
        delta[..., :self.model.encoder.emb_dim].uniform_(-visual_eps, visual_eps)
        delta[..., self.model.encoder.emb_dim:-self.model.action_dim].uniform_(-proprio_eps, proprio_eps)
        delta[..., -self.model.action_dim:].uniform_(-action_eps, action_eps)
        delta.requires_grad_()

        _, _, _, _, loss, _ = self.model._forward(obs, act, z, perturbation=delta)

        (g_delta,) = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)
        with torch.no_grad():
            delta[..., :self.model.encoder.emb_dim] += visual_alpha * g_delta[..., :self.model.encoder.emb_dim].sign()
            delta[..., :self.model.encoder.emb_dim].clamp_(-visual_eps, visual_eps)

            delta[..., self.model.encoder.emb_dim:-self.model.action_dim] += propio_alpha * g_delta[..., self.model.encoder.emb_dim:-self.model.action_dim].sign()
            delta[..., self.model.encoder.emb_dim:-self.model.action_dim].clamp_(-proprio_eps, proprio_eps)

            delta[..., -self.model.action_dim:] += action_alpha * g_delta[..., -self.model.action_dim:].sign()
            delta[..., -self.model.action_dim:].clamp_(-action_eps, action_eps)
        # ----------------------------------------------------------------------

        # Gradient Descent
        # ----------------------------------------------------------------------
        self.encoder_optimizer.zero_grad()
        if self.cfg.has_decoder:
            self.decoder_optimizer.zero_grad()
        if self.cfg.has_predictor:
            self.predictor_optimizer.zero_grad()
            self.action_encoder_optimizer.zero_grad()

        delta = delta.detach()
        z_out, visual_out, visual_reconstructed, unperturbed_visual_reconstructed, loss, loss_components = self.model(
            obs, act, perturbation=delta
        )

        self.accelerator.backward(loss)

        if self.model.train_encoder:
            self.encoder_optimizer.step()
        if self.cfg.has_decoder and self.model.train_decoder:
            self.decoder_optimizer.step()
        if self.cfg.has_predictor and self.model.train_predictor:
            self.predictor_optimizer.step()
            self.action_encoder_optimizer.step()
        # ----------------------------------------------------------------------

        return loss, loss_components, visual_eps, proprio_eps, action_eps, visual_out, visual_reconstructed, unperturbed_visual_reconstructed

    def train_adversarial_wm(self):
        for i, data in enumerate(
            tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train Adversarial WM")
        ):
            self.train_step += 1
            _, obs, act, state, goal, T = data
            plot = (i % 10 == 0) and self.accelerator.is_main_process
            self.model.train()

            act = act.unfold(1, self.num_frames, 1) # (B, N, action_dim, num_frames)
            act = rearrange(act, "B N action_dim num_frames -> (B N) num_frames action_dim")
            obs = {"visual": rearrange(obs['visual'], "b n f c h w -> (b n) f c h w"), "proprio": rearrange(obs['proprio'], "b n f proprio_dim -> (b n) f proprio_dim")}

            loss, loss_components, visual_eps, proprio_eps, action_eps, visual_out, visual_reconstructed, unperturbed_visual_reconstructed = self.fgsm_step(obs, act, self.cfg.visual_eps_factor, self.cfg.proprio_eps_factor, self.cfg.action_eps_factor)

            loss = self.accelerator.gather_for_metrics(loss).mean()
            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }

            self.logs_update({
                "train/wm_loss": [loss_components["z_loss"]],
                "train/decoder_loss": [loss_components["decoder_loss_reconstructed"]],
                "train/loss": [loss.item()],
                "train/visual_eps": [visual_eps.item()],
                "train/proprio_eps": [proprio_eps.item()],
                "train/action_eps": [action_eps.item()],
            }, phase="train")

            if self.cfg.has_decoder and plot:
                # print("train", obs["visual"].shape, visual_out.shape, visual_reconstructed.shape)
                # obs["visual"]: (80,      4, 3, 224, 224)
                #                (B*(T-3), 4, 3, 224, 224)

                # visual_out:   (80,      3, 3, 224, 224)
                #               (B*(T-3), 3, 3, 224, 224)

                # visual_reconstructed: (80,      4, 3, 224, 224)
                #                       (B*(T-3), 4, 3, 224, 224)
                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="train",
                    unperturbed_reconstructed_imgs=unperturbed_visual_reconstructed,
                )


    def train_wm(self):
        for i, data in enumerate(
            tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train WM")
        ):
            self.train_step += 1
            obs, act, state, goal, T = data
            plot = i % self.cfg.training.save_every_x_batch == 0
            self.model.train()

            act = act.unfold(1, self.num_frames, 1) # (B, N, action_dim, num_frames)
            act = rearrange(act, "B N action_dim num_frames -> (B N) num_frames action_dim")
            obs = {"visual": rearrange(obs['visual'], "b n f c h w -> (b n) f c h w"), "proprio": rearrange(obs['proprio'], "b n f proprio_dim -> (b n) f proprio_dim")}
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act
            )

            self.encoder_optimizer.zero_grad()
            if self.cfg.has_decoder:
                self.decoder_optimizer.zero_grad()
            if self.cfg.has_predictor:
                self.predictor_optimizer.zero_grad()
                self.action_encoder_optimizer.zero_grad()

            self.accelerator.backward(loss)

            if self.model.train_encoder:
                self.encoder_optimizer.step()
            if self.cfg.has_decoder and self.model.train_decoder:
                self.decoder_optimizer.step()
            if self.cfg.has_predictor and self.model.train_predictor:
                self.predictor_optimizer.step()
                self.action_encoder_optimizer.step()
        
            loss = self.accelerator.gather_for_metrics(loss).mean()
            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }

            self.logs_update({
                "train/wm_loss": [loss_components["z_loss"]],
                "train/decoder_loss": [loss_components["decoder_loss_reconstructed"]],
                "train/loss": [loss.item()],
            }, phase="train")

            if self.cfg.has_decoder and plot:
                print("train", obs["visual"].shape, visual_out.shape, visual_reconstructed.shape)
                # obs["visual"]: (80,      4, 3, 224, 224)
                #                (B*(T-3), 4, 3, 224, 224)

                # visual_out:   (80,      3, 3, 224, 224)
                #               (B*(T-3), 3, 3, 224, 224)

                # visual_reconstructed: (80,      4, 3, 224, 224)
                #                       (B*(T-3), 4, 3, 224, 224)
                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="train",
                )


    def train(self):
        for i, data in enumerate(
            tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train")
        ):
            self.train_step += 1
            obs, act, state, goal, T = data
            plot = i % 10 == 0
            self.model.train()

            # inner loop
            if self.cfg.has_innerloop:
                obs_0 = {"visual": obs['visual'][:, 0, :, :, :, :], "proprio": obs['proprio'][:, 0, :, :]}
                rollout_goal = {"visual": obs['visual'][:, -1, :, :, :, :], "proprio": obs['proprio'][:, -1, :, :]}
                goal = {"visual": goal['visual'].unsqueeze(1), "proprio": goal['proprio'].unsqueeze(1)}
                z_rollout_goal = self.model.encode_z(rollout_goal)[:, -1, :, :]
                z_goal = self.model.encode_z(goal)[:, 0, :, :]
                z_0 = self.model.encode_z(obs_0)[:, 0, :, :]
                act_hat = self.initnet(z_0, z_goal)
                grad_act = torch.zeros_like(act_hat)
                # add momentum back later
                #old_grad_act = torch.zeros_like(act_hat)
                for i in range(self.cfg.inner_iters):
                    z_obses, _, _ = self.model.rollout(obs_0, act_hat)
                    loss = self.model.emb_criterion(z_obses[:, -1, ...], z_rollout_goal)
                    grads = torch.autograd.grad([loss], [act_hat], create_graph=True, allow_unused=False)[0]
                    assert grads is not None
                    grad_act = grads
                    act_hat = act_hat - self.cfg.training.inner_lr * (grad_act)
                    #old_grad_act = grad_act.detach()

                act_emb = self.model.encode_act(act)
                # need to add regularization back
                actions_loss = (1 - self.cfg.inner_scale) * self.model.emb_criterion(act_emb, act_hat)

            act = act.unfold(1, self.num_frames, 1) # (B, N, action_dim, num_frames)
            act = act.permute(1, 0, 3, 2) # (N, B, num_frames, action_dim)
            obs = {"visual": obs['visual'].transpose(0, 1), "proprio": obs['proprio'].transpose(0, 1)}
            wm_loss = 0
            visual_outs = []
            visual_reconstructeds = []
            for i in range(self.cfg.N):
                obs_i = {"visual": obs['visual'][i], "proprio": obs['proprio'][i]}
                z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                    obs_i, act[i]
                )
                wm_loss += loss
                visual_outs.append(visual_out)  
                visual_reconstructeds.append(visual_reconstructed)

            wm_loss = (self.cfg.inner_scale) * wm_loss / self.cfg.N

            loss = wm_loss + actions_loss if self.cfg.has_innerloop else wm_loss
            self.encoder_optimizer.zero_grad()
            if self.cfg.has_decoder:
                self.decoder_optimizer.zero_grad()
            if self.cfg.has_predictor:
                self.predictor_optimizer.zero_grad()
                self.action_encoder_optimizer.zero_grad()
            if self.cfg.has_innerloop:
                self.initnet_optimizer.zero_grad()

            self.accelerator.backward(loss)

            if self.model.train_encoder:
                self.encoder_optimizer.step()
            if self.cfg.has_decoder and self.model.train_decoder:
                self.decoder_optimizer.step()
            if self.cfg.has_predictor and self.model.train_predictor:
                self.predictor_optimizer.step()
                self.action_encoder_optimizer.step()
            if self.cfg.has_innerloop:
                self.initnet_optimizer.step()
        
            loss = self.accelerator.gather_for_metrics(loss).mean()
            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }

            self.logs_update({
                "train/wm_loss": [loss_components["z_loss"]/self.cfg.inner_scale],
                "train/decoder_loss": [loss_components["decoder_loss_reconstructed"]/self.cfg.inner_scale],
                "train/actions_loss": [actions_loss/(1-self.cfg.inner_scale)],
                "train/loss": [loss.item()],
            }, phase="train")

            
            if self.cfg.has_decoder and plot:

                self.plot_samples(
                    obs["visual"],
                    visual_outs,
                    visual_reconstructeds,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="train",
                )

            # loss_components = {f"train_{k}": [v] for k, v in loss_components.items()}
            # self.logs_update(loss_components, step=self.train_step)


    def val(self):
        self.model.eval()

        self.accelerator.wait_for_everyone()
        for i, data in enumerate(
            tqdm(self.dataloaders["valid"], desc=f"Epoch {self.epoch} Valid")
        ):
            self.val_step += 1
            obs, act, state, goal, T = data
            plot = i % 10 == 0
            self.model.eval()

            # only use first batch for now
            obs = {k: v[:1] for k, v in obs.items()}
            act = act[:1]
            state = state[:1]
            goal = {k: v[:1] for k, v in goal.items()}

            # inner loop
            obs_0 = {"visual": obs['visual'][:, 0, :, :, :, :], "proprio": obs['proprio'][:, 0, :, :]}
            obs_goal = {"visual": obs['visual'][:, -1, :, :, :, :], "proprio": obs['proprio'][:, -1, :, :]}
            z_goal = self.model.encode_z(obs_goal)[:, -1, :, :]
            z_0 = self.model.encode_z(obs_0)[:, 0, :, :]
            z_start = None
            visual_preds = []
            t = self.cfg.num_hist
            act_hats = []
            while t < self.cfg.val_rollout_steps + self.cfg.num_hist:
                
                rollout_goal = {"visual": obs['visual'][:, t - self.cfg.num_hist + self.cfg.N - 1, :, :, :, :], "proprio": obs['proprio'][:, t - self.cfg.num_hist + self.cfg.N - 1, :, :]}
                z_rollout_goal = self.model.encode_z(rollout_goal)[:, -1, :, :]
                
                act_hat = self.initnet(z_0, z_goal)
                grad_act = torch.zeros_like(act_hat)
                # add momentum back later
                #old_grad_act = torch.zeros_like(act_hat)
                for i in range(self.cfg.inner_iters):
                    z_obses, z, visual_pred = self.model.rollout(obs_0, act_hat, z_start=z_start)
                    
                    loss = self.model.emb_criterion(z_obses[:, -1, ...], z_rollout_goal)
                    grads = torch.autograd.grad([loss], [act_hat],create_graph=False, allow_unused=False)[0]
                    assert grads is not None
                    grad_act = grads
                    act_hat = act_hat - self.cfg.training.inner_lr * (grad_act)
                    #old_grad_act = grad_act.detach()
                if t > self.cfg.num_hist:
                    act_hats.append(act_hat[:, self.cfg.num_hist:])
                else:
                    act_hats.append(act_hat)

                visual_preds.append(visual_pred)
                z_0 = z_obses[:, -self.cfg.num_hist, ...]
                z_start = z[:, -self.cfg.num_hist:, ...]
                t += self.cfg.N

            acts_hat = torch.cat(act_hats, dim=1)
            acts_hat = acts_hat[:, self.cfg.num_hist:]
            acts_hat = rearrange(acts_hat, "b t (f d) -> b (t f) d", f=self.cfg.frameskip)
            acts_hat = self.val_data_preprocessor.denormalize_actions(acts_hat.detach().cpu())

            act = act[:, self.cfg.num_hist:self.cfg.num_hist + self.cfg.val_rollout_steps]
            act = rearrange(act, "b t (f d) -> b (t f) d", f=self.cfg.frameskip)
            act = self.val_data_preprocessor.denormalize_actions(act.detach().cpu())
            
            acts_hat = acts_hat.squeeze(0).numpy()
            act = act.squeeze(0).numpy()
            init_state = state[0, self.cfg.num_hist].detach().cpu().numpy()
            rollout_obs_hat, rollout_states_hat = self.env.rollout(self.cfg.training.seed, init_state, acts_hat)
            rollout_obs, rollout_states = self.env.rollout(self.cfg.training.seed, init_state, act)

            # don't include last state because doesn't align with frameskip
            rollout_obs_hat = {k: v[:-1] for k, v in rollout_obs_hat.items()}
            rollout_states_hat = rollout_states_hat[:-1]
            visual_preds = rollout_obs_hat["visual"]

            rollout_obs = {k: v[:-1] for k, v in rollout_obs.items()}
            rollout_states = rollout_states[:-1]
            visual_gt = rollout_obs["visual"]
            
            goal_state = state[0, -1].detach().cpu().numpy()
            val_goal_state_loss = np.linalg.norm(rollout_states[-1] - goal_state)

            self.plot_goal_state(visual_preds[-1])
                
            self.plot_samples(
                None,
                visual_gt,
                None,
                self.epoch,
                batch=i,
                num_samples=self.num_reconstruct_samples,
                phase="val",
                title="gt"
            )
            self.plot_samples(
                None,
                visual_preds,
                None,
                self.epoch,
                batch=i,
                num_samples=self.num_reconstruct_samples,
                phase="val",
                title="pred"
            )
            self.logs_update({"val/goal_state_loss": [val_goal_state_loss.item()]}, phase="val")


    def openloop_rollout(
        self, dset, num_rollout=10, rand_start_end=True, min_horizon=2, mode="train"
    ):
        np.random.seed(self.cfg.training.seed)
        min_horizon = min_horizon + self.cfg.num_hist
        plotting_dir = f"rollout_plots/e{self.epoch}_rollout"
        if self.accelerator.is_main_process:
            os.makedirs(plotting_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        logs = {}

        # rollout with both num_hist and 1 frame as context
        num_past = [(self.cfg.num_hist, ""), (1, "_1framestart")]

        # sample traj
        for idx in range(num_rollout):
            valid_traj = False
            while not valid_traj:
                traj_idx = np.random.randint(0, len(dset))
                obs, act, state, _ = dset[traj_idx]
                act = act.to(self.device)
                if rand_start_end:
                    if obs["visual"].shape[0] > min_horizon * self.cfg.frameskip + 1:
                        start = np.random.randint(
                            0,
                            obs["visual"].shape[0] - min_horizon * self.cfg.frameskip - 1,
                        )
                    else:
                        start = 0
                    max_horizon = (obs["visual"].shape[0] - start - 1) // self.cfg.frameskip
                    if max_horizon > min_horizon:
                        valid_traj = True
                        horizon = np.random.randint(min_horizon, max_horizon + 1)
                else:
                    valid_traj = True
                    start = 0
                    horizon = (obs["visual"].shape[0] - 1) // self.cfg.frameskip

            for k in obs.keys():
                obs[k] = obs[k][
                    start : 
                    start + horizon * self.cfg.frameskip + 1 : 
                    self.cfg.frameskip
                ]
            act = act[start : start + horizon * self.cfg.frameskip]
            act = rearrange(act, "(h f) d -> h (f d)", f=self.cfg.frameskip)

            obs_g = {}
            for k in obs.keys():
                obs_g[k] = obs[k][-1].unsqueeze(0).unsqueeze(0).to(self.device)
            z_g = self.model.encode_obs(obs_g)
            actions = act.unsqueeze(0)

            for past in num_past:
                n_past, postfix = past

                obs_0 = {}
                for k in obs.keys():
                    obs_0[k] = (
                        obs[k][:n_past].unsqueeze(0).to(self.device)
                    )  # unsqueeze for batch, (b, t, c, h, w)

                z_obses, z, _ = self.model.rollout(obs_0, actions)
                z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)
                div_loss = self.err_eval_single(z_obs_last, z_g)

                for k in div_loss.keys():
                    log_key = f"z_{k}_err_rollout{postfix}"
                    if log_key in logs:
                        logs[f"z_{k}_err_rollout{postfix}"].append(
                            div_loss[k]
                        )
                    else:
                        logs[f"z_{k}_err_rollout{postfix}"] = [
                            div_loss[k]
                        ]

                if self.cfg.has_decoder:
                    visuals = self.model.decode_obs(z_obses)[0]["visual"]
                    imgs = torch.cat([obs["visual"], visuals[0].cpu()], dim=0)
                    self.plot_imgs(
                        imgs,
                        obs["visual"].shape[0],
                        f"{plotting_dir}/e{self.epoch}_{mode}_{idx}{postfix}.png",
                    )
        logs = {
            key: sum(values) / len(values) for key, values in logs.items() if values
        }
        return logs

    def logs_update(self, logs, phase, log=True):
        step_log = {}
        for key, value in logs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            length = len(value)
            count, total = self.epoch_log.get(key, (0, 0.0))
            self.epoch_log[key] = (
                count + length,
                total + sum(value),
            )
            step_log[key + "_step"] = value[-1]
        # include the appropriate phase step metric for define_metric routing
        if phase == "pretrain_initnet":
            step_log["pretrain_initnet_step"] = self.pretrain_initnet_step
        elif phase == "train":
            step_log["train_step"] = self.train_step
        elif phase == "val":
            step_log["val_step"] = self.val_step
        if self.accelerator.is_main_process and log:
            self.wandb_run.log(step_log)

    def logs_flash(self):
        epoch_log = OrderedDict()
        for key, value in self.epoch_log.items():
            count, sum = value
            to_log = sum / count
            epoch_log[key + "_epoch"] = to_log
        epoch_log["epoch"] = self.epoch
        log.info(f"Epoch {self.epoch}  Training loss: {epoch_log['train_loss_epoch']:.4f}  \
                Validation loss: {epoch_log['val_loss_epoch']:.4f}")

        if self.accelerator.is_main_process:
            # include both step metrics so phase groups route correctly
            epoch_log_with_steps = dict(epoch_log)
            epoch_log_with_steps["train_step"] = self.train_step
            epoch_log_with_steps["val_step"] = self.val_step
            self.wandb_run.log(epoch_log_with_steps)
        self.epoch_log = OrderedDict()

    def plot_samples(
        self,
        gt_imgs,
        pred_imgs,
        reconstructed_gt_imgs,
        epoch,
        batch,
        num_samples=2,
        phase="train",
        unperturbed_reconstructed_imgs=None,
    ):
        """
        input:
            gt_imgs: (N, B, T, 3, H, W) or (B, T, 3, H, W)
            pred_imgs: list of length N of (B, H, 3, H, W) or tensor (B, H, 3, H, W)
            reconstructed_gt_imgs: list of length N of (B, T, 3, H, W) or tensor (B, T, 3, H, W)
        output: PNG grid and three MP4s (gt, pred, recon)
        """
        if gt_imgs is None and reconstructed_gt_imgs is None:
            pred_imgs = pred_imgs[:, 0, :, :, :]
            pred_imgs = self.normalize_video(pred_imgs) * 255.0
            log_payload = {
                f"{phase}/pred": wandb.Video(pred_imgs.detach().cpu().numpy().astype(np.uint8), fps=4, format="gif"),
            }
            # add phase step
            if phase == "pretrain_initnet":
                log_payload["pretrain_initnet_step"] = self.pretrain_initnet_step
            elif phase == "train":
                log_payload["train_step"] = self.train_step
            elif phase == "val":
                log_payload["val_step"] = self.val_step
            self.wandb_run.log(log_payload)
            return
        if len(gt_imgs.shape) == 5:
            gt_imgs = rearrange(gt_imgs, "(n b) f c h w -> n b f c h w", b=self.cfg.training.batch_size)
            pred_imgs = rearrange(pred_imgs, "(n b) f c h w -> n b f c h w", b=self.cfg.training.batch_size)
            reconstructed_gt_imgs = rearrange(reconstructed_gt_imgs, "(n b) f c h w -> n b f c h w", b=self.cfg.training.batch_size)
        
        gt_imgs = gt_imgs[:, 0, -1, :, :, :] # (n, num_frames, 3, img_size, img_size)
        if type(pred_imgs) == list:
            pred_imgs = torch.stack(pred_imgs) # (n, num_hist, 3, img_size, img_size)
        pred_imgs = pred_imgs[:, 0, -1, :, :, :]
        if type(reconstructed_gt_imgs) == list:
            reconstructed_gt_imgs = torch.stack(reconstructed_gt_imgs) 
        reconstructed_gt_imgs = reconstructed_gt_imgs[:, 0, -1, :, :, :]

        gt_imgs = self.normalize_video(gt_imgs) * 255.0
        pred_imgs = self.normalize_video(pred_imgs) * 255.0
        reconstructed_gt_imgs = self.normalize_video(reconstructed_gt_imgs) * 255.0

        log_payload = {
            f"{phase}/gt": wandb.Video(gt_imgs.detach().cpu().numpy().astype(np.uint8), fps=4, format="gif"),
            f"{phase}/pred": wandb.Video(pred_imgs.detach().cpu().numpy().astype(np.uint8), fps=4, format="gif"),
            f"{phase}/reconstructed": wandb.Video(reconstructed_gt_imgs.detach().cpu().numpy().astype(np.uint8), fps=4, format="gif"),
        }

        if unperturbed_reconstructed_imgs is not None:
            unperturbed_reconstructed_imgs = rearrange(unperturbed_reconstructed_imgs, "(n b) f c h w -> n b f c h w", b=self.cfg.training.batch_size)
            if type(unperturbed_reconstructed_imgs) == list:
                unperturbed_reconstructed_imgs = torch.stack(unperturbed_reconstructed_imgs)
            unperturbed_reconstructed_imgs = unperturbed_reconstructed_imgs[:, 0, -1, :, :, :]

            unperturbed_reconstructed_imgs = self.normalize_video(unperturbed_reconstructed_imgs) * 255.0
            log_payload[f"{phase}/unperturbed_reconstructed"] = wandb.Video(unperturbed_reconstructed_imgs.detach().cpu().numpy().astype(np.uint8), fps=4, format="gif")

        if phase == "pretrain_initnet":
            log_payload["pretrain_initnet_step"] = self.pretrain_initnet_step
        elif phase == "train":
            log_payload["train_step"] = self.train_step
        elif phase == "val":
            log_payload["val_step"] = self.val_step
        self.wandb_run.log(log_payload)

    def normalize_video(self, video):
        tensor = video.clone()

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            norm_ip(t, range[0], range[1])

        norm_range(tensor, (0, 1))
        return tensor

        


@hydra.main(config_path="conf", config_name="train")
def main(cfg: OmegaConf):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
