import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat
from collections import defaultdict

class VWorldModel(nn.Module):
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        decoder,
        predictor,
        initnet,
        proprio_dim=0,
        action_dim=0,
        concat_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        train_initnet=False,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.decoder = decoder  # decoder could be None
        self.predictor = predictor  # predictor could be None
        self.initnet = initnet  # initnet could be None
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.train_initnet = train_initnet
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.proprio_dim = proprio_dim * num_proprio_repeat 
        self.action_dim = action_dim * num_action_repeat 
        self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim) # Not used

        print(f"num_action_repeat: {self.num_action_repeat}")
        print(f"num_proprio_repeat: {self.num_proprio_repeat}")
        print(f"proprio encoder: {proprio_encoder}")
        print(f"action encoder: {action_encoder}")
        print(f"proprio_dim: {proprio_dim}, after repeat: {self.proprio_dim}")
        print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        print(f"emb_dim: {self.emb_dim}")

        self.concat_dim = concat_dim # 0 or 1
        assert concat_dim == 0 or concat_dim == 1, f"concat_dim {concat_dim} not supported."
        print("Model emb_dim: ", self.emb_dim)

        if "dino" in self.encoder.name:
            decoder_scale = 16  # from vqvae
            num_side_patches = image_size // decoder_scale
            self.encoder_image_size = num_side_patches * encoder.patch_size
            self.encoder_transform = transforms.Compose(
                [transforms.Resize(self.encoder_image_size)]
            )
        else:
            # set self.encoder_transform to identity transform
            self.encoder_transform = lambda x: x

        self.decoder_criterion = nn.MSELoss()
        self.decoder_latent_loss_weight = 0.25
        self.emb_criterion = nn.MSELoss()

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.predictor is not None and self.train_predictor:
            self.predictor.train(mode)
        self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)
        if self.decoder is not None and self.train_decoder:
            self.decoder.train(mode)
        if self.initnet is not None:
            self.initnet.train(mode)
    
    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        self.proprio_encoder.eval()
        self.action_encoder.eval()
        if self.decoder is not None:
            self.decoder.eval() 
        if self.initnet is not None:
            self.initnet.eval()

    def encode(self, obs, act): 
        """
        input :  obs (dict): "visual", "proprio", (b, num_frames, 3, img_size, img_size) 
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z = torch.cat(
                    [z_dct['visual'], z_dct['proprio'].unsqueeze(2), act_emb.unsqueeze(2)], dim=2 # add as an extra token
                )  # (b, num_frames, num_patches + 2, dim)
        if self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z = torch.cat(
                [z_dct['visual'], proprio_repeated, act_repeated], dim=3
            )  # (b, num_frames, num_patches, dim + action_dim)
        return z
    
    def encode_act(self, act):
        act = self.action_encoder(act) # (b, num_frames, action_emb_dim)
        return act
    
    def encode_proprio(self, proprio):
        proprio = self.proprio_encoder(proprio)
        return proprio

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        visual = obs['visual']
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        visual_embs = self.encoder.forward(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)

        proprio = obs['proprio']
        proprio_emb = self.encode_proprio(proprio)
        return {"visual": visual_embs, "proprio": proprio_emb}

    def encode_z(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output: z : (b, t, num_patches, encoder_emb_dim + proprio_dim)
        """
        z_dct = self.encode_obs(obs)
        if self.concat_dim == 0:
            z = torch.cat(
                    [z_dct['visual'], z_dct['proprio'].unsqueeze(2)], dim=2 # add as an extra token
                )  # (b, num_frames, num_patches + 2, dim + proprio_dim)
        if self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            z = torch.cat(
                [z_dct['visual'], proprio_repeated], dim=3
            )  # (b, num_frames, num_patches, dim + proprio_dim)
        return z

    def predict(self, z):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        T = z.shape[1]
        # reshape to a batch of windows of inputs
        z = rearrange(z, "b t p d -> b (t p) d")
        # (b, num_hist * num_patches per img, emb_dim)
        z = self.predictor(z)
        z = rearrange(z, "b (t p) d -> b t p d", t=T)
        return z

    def decode(self, z):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        z_obs, z_act = self.separate_emb_to_dict(z)
        obs, diff = self.decode_obs(z_obs)
        return obs, diff

    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        b, num_frames, num_patches, emb_dim = z_obs["visual"].shape
        visual, diff = self.decoder(z_obs["visual"])  # (b*num_frames, 3, 224, 224)
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
        obs = {
            "visual": visual,
            "proprio": z_obs["proprio"], # Note: no decoder for proprio for now!
        }
        return obs, diff
    
    def separate_emb_to_dict(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """
        if self.concat_dim == 0:
            z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim + self.action_dim)], \
                                         z[..., -(self.proprio_dim + self.action_dim) :-self.action_dim],  \
                                         z[..., -self.action_dim:]
            # remove tiled dimensions
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs, z_act

    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """
        if self.concat_dim == 0:
            z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_obs, z_act = z[..., :-(self.action_dim)],z[..., -self.action_dim:]
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        return z_obs, z_act

    def concat_emb(self, z_emb, act_emb):
        if self.concat_dim == 0:
            z = torch.cat(
                    [z_emb, act_emb.unsqueeze(2)], dim=2 # add as an extra token
                )  # (b, num_frames, num_patches + 2, dim)
        if self.concat_dim == 1:
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_emb.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z = torch.cat(
                [z_emb, act_repeated], dim=3
            )
        return z
    
    def forward(self, obs, act, perturbation=None):
        z = self.encode(obs, act)
        return self._forward(obs, act, z, perturbation)

    def _forward(self, obs, act, z, perturbation=None):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                visual_pred: (b, num_hist, 3, img_size, img_size)
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        loss = 0
        loss_components = {}
        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)

        if perturbation is not None:
            z_src = z_src + perturbation

        z_tgt = z[:, self.num_pred :, :, :]  # (b, num_hist, num_patches, dim)
        visual_src = obs['visual'][:, : self.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
        visual_tgt = obs['visual'][:, self.num_pred :, ...]  # (b, num_hist, 3, img_size, img_size)

        if self.predictor is not None:
            z_pred = self.predict(z_src)
            if self.decoder is not None:
                obs_pred, diff_pred = self.decode(
                    z_pred.detach()
                )  # recon loss should only affect decoder
                visual_pred = obs_pred['visual']
                recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                decoder_loss_pred = (
                    recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            # Compute loss for visual, proprio dims (i.e. exclude action dims)
            if self.concat_dim == 0:
                z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt[:, :, :-2, :].detach())
                z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt[:, :, -2, :].detach())
                z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
            elif self.concat_dim == 1:
                z_visual_loss = self.emb_criterion(
                    z_pred[:, :, :, :-(self.proprio_dim + self.action_dim)], \
                    z_tgt[:, :, :, :-(self.proprio_dim + self.action_dim)].detach()
                )
                z_proprio_loss = self.emb_criterion(
                    z_pred[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim], 
                    z_tgt[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim].detach()
                )
                z_loss = self.emb_criterion(
                    z_pred[:, :, :, :-self.action_dim], 
                    z_tgt[:, :, :, :-self.action_dim].detach()
                )

            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss
        else:
            visual_pred = None
            z_pred = None

        if self.decoder is not None:
            obs_reconstructed, diff_reconstructed = self.decode(
                z.detach()
            )  # recon loss should only affect decoder
            visual_reconstructed = obs_reconstructed["visual"]

            if perturbation is not None:
                unperturbed_obs_reconstructed, _ = self.decode(
                    (z_src - perturbation).detach()
                )
                unperturbed_visual_reconstructed = unperturbed_obs_reconstructed["visual"]

            recon_loss_reconstructed = self.decoder_criterion(visual_reconstructed, obs['visual'])
            decoder_loss_reconstructed = (
                recon_loss_reconstructed
                + self.decoder_latent_loss_weight * diff_reconstructed
            )

            loss_components["decoder_recon_loss_reconstructed"] = (
                recon_loss_reconstructed
            )
            loss_components["decoder_vq_loss_reconstructed"] = diff_reconstructed
            loss_components["decoder_loss_reconstructed"] = (
                decoder_loss_reconstructed
            )
            loss = loss + decoder_loss_reconstructed
        else:
            visual_reconstructed = None
        loss_components["loss"] = loss

        if perturbation is not None:
            return z_pred, visual_pred, visual_reconstructed, unperturbed_visual_reconstructed, loss, loss_components
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    def replace_actions_from_z(self, z, act, embed=True):
        if embed:
            act_emb = self.encode_act(act)
        else:
            act_emb = act
        if self.concat_dim == 0:
            z[:, :, -1, :] = act_emb
        elif self.concat_dim == 1:
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z[..., -self.action_dim:] = act_repeated
        return z

    def rollout_legacy(self, obs_0, act):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """
        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:] 
        z = self.encode(obs_0, act_0)
        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred = self.predict(z[:, -self.num_hist :])
            z_new = z_pred[:, -inc:, ...]
            z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        z_obses, z_acts = self.separate_emb_to_dict(z)
        return z_obses, z

    def rollout(self, obs_0, act_emb, z_start=None, detach=True, track_grads=False):
        """
        Returns: z_obses, z, visual_preds
        Side-effect (if track_grads=True): fills self._tbptt_grad_log with per-t grad norms.
        """
        if track_grads:
            self._tbptt_grad_log = defaultdict(list)  # keys: 'z_in', 'z_new' -> list[(t, norm)]

        act_0 = act_emb[:, :self.num_hist]
        action = act_emb[:, self.num_hist:]

        if z_start is not None:
            z = z_start
        else:
            z = self.encode_z(obs_0)[:, :self.num_hist, :, :]
            z = self.concat_emb(z, act_0)

        visual_preds = []
        t, T = 0, action.shape[1]
        while t < T:
            z_in = z[:, -self.num_hist:]  # view; part of graph

            if track_grads and z_in.requires_grad:
                def make_hook(tt):
                    def _hook(grad):
                        # store L2 norm; use .detach() to avoid any autograd linkage
                        self._tbptt_grad_log['z_in'].append((tt, grad.norm().detach().item()))
                    return _hook
                z_in.register_hook(make_hook(t))

            z_pred = self.predict(z_in)

            if self.decoder is not None:
                obs_pred, diff_pred = self.decode(z_pred.detach())
                visual_pred = obs_pred['visual'][:, -1, ...]
                visual_preds.append(visual_pred)

            z_new = z_pred[:, -1:, ...]
            if track_grads and z_new.requires_grad:
                def make_hook_new(tt):
                    def _hook(grad):
                        self._tbptt_grad_log['z_new'].append((tt, grad.norm().detach().item()))
                    return _hook
                z_new.register_hook(make_hook_new(t))

            if t < T - 1:
                z_new = self.replace_actions_from_z(z_new, action[:, t:t+1, :], embed=False)

            z = torch.cat([z, z_new], dim=1)
            t += 1

        z_obses, z_acts = self.separate_emb(z)
        visual_preds = torch.stack(visual_preds) if len(visual_preds) else None
        return z_obses, z, visual_preds
    
    def custom_rollout(self, obs_0, actions, z_gt=None):
        """
        Does wm rollouts using last latent only (and not h)

        Params:
        -------
            z0: tensor of size (bs, self.visual_dim + self.proprio_dim)
            actions: tensor of size (bs, T, action_dim)
        Returns:
        --------
            pred_dict: dict with keys 'visual' and 'proprio'
                pred_dict['visual']: tensor of size (bs, T + 1, self.h, self.w)  # h and w are the dino sized
                pred_dict['proprio']: tensor of size (bs, T + 1, self.proprio_dim)
        """
        z_obs_dict = self.encode_obs(obs_0)
        z0 = self.flatten_z_from_dict(z_obs_dict, init=True)
        bs = z0.shape[0]
        device = z0.device
        _, T, *_ = actions.shape
        z_dim = z0.shape[1]
        z_preds = torch.zeros(bs, T + 1, z_dim).to(device)
        z_preds[:, 0] = z0
        for t in range(T):
            z_pred = self.pred_next(z_preds[:, t], actions[:, t])
            z_preds[:, t+1] = z_pred
        pred_dict = self.make_pred_dict_from_zs(z_preds)
        
        return pred_dict
    
    def flatten_z_from_dict(self, z_dict, init=False):
        """
        Params:
        -------
            z_dict: dict with keys 'visual' and 'proprio'
                pred_dict['visual']: tensor of size (bs, 1, self.h, self.w)  # h and w are the dino sized
                pred_dict['proprio']: tensor of size (bs, 1, self.proprio_dim)
            init: bool
                If True, will save h, w and visual dim
                    
        Returns:
        --------
            z_flat: tensor of size (bs, self.visual_dim + self.proprio_dim)
        """
        bs, *_ = z_dict['visual'].shape
        z_visual_flat = z_dict['visual'].reshape((bs, -1))
        z_proprio_flat = z_dict['proprio'].reshape((bs, -1))
        z_flat = torch.cat((z_visual_flat, z_proprio_flat), dim=1)
        if init:
            self.h, self.w = z_dict['visual'].shape[2], z_dict['visual'].shape[3]
            self.visual_dim = z_visual_flat.shape[1]
        return z_flat
    
    def make_pred_dict_from_zs(self, z_preds):
        """
        Params:
        -------
            z_preds: tensor of size (bs, T, self.visual_dim + self.proprio_dim)
        Returns:
        --------
            pred_dict: dict with keys 'visual' and 'proprio'
                pred_dict['visual']: tensor of size (bs, T, self.h, self.w)  # h and w are the dino sized
                pred_dict['proprio']: tensor of size (bs, T, self.proprio_dim)
        """
        bs = z_preds.shape[0]
        T = z_preds.shape[1]
        pred_dict = dict()
        pred_dict['visual'] = z_preds[:, :, :self.visual_dim].view((bs, T, self.h, self.w))
        pred_dict['proprio'] = z_preds[:, :, self.visual_dim:self.visual_dim + self.proprio_dim].view((bs, T, self.proprio_dim))

        return pred_dict

    def pred_next(self, z, u):
            """
            Params:
            -------
                z: vector of dim (bs, self.visual_dim + self.proprio_dim)
                u: vector of dim (bs, action_dim)
            Output:
            -------
                concat: vector of dim (bs, self.visual_dim + self.proprio_dim)
            """
            z = self.reshape_state(z)
            act = self.reshape_action(u)
            zu = torch.cat([z, act], dim=3)
            z_pred = self.predict(zu)[:, :, :, :]
            concat = self.reshape_pred(z_pred)

            return concat

    def reshape_state(self, z):
            """
            Params:
            -------
                z: vector of size (bs, self.visual_dim + self.proprio_dim)

            Returns:
            --------
                z: reshaped vector of size (bs, ...?)
            """
            bs = z.shape[0]
            z_visual_flat = z[:, :self.visual_dim]
            z_visual = z_visual_flat.view((bs, 1, self.h, self.w))
            z_proprio = z[:, self.visual_dim:self.visual_dim + self.proprio_dim].view((bs, 1, self.proprio_dim))
            proprio_tiled = repeat(z_proprio.unsqueeze(2), "b t 1 a -> b t f a", f=self.h)
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            z = torch.cat([z_visual, proprio_repeated], dim=3)

            return z
    
    def reshape_action(self, u):
        """
        Params:
        -------
            u: vector of dim ?

        Returns:
        --------
            u_reshaped: vector of shape?
        """
        act_emb = self.encode_act(u)
        act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=self.h)
        act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
        
        return act_repeated
    
    def reshape_pred(self, z_pred):
        """
        Params:
        -------
            z_pred: vector of dim ?
        Output:
        -------
            z: vector of dim (bs, self.visual_dim + self.proprio_dim)
        """
        bs = z_pred.shape[0]
        z_visual, z_proprio, z_act = z_pred[..., :-(self.proprio_dim + self.action_dim)],z_pred[..., -(self.proprio_dim + self.action_dim) :-self.action_dim],z_pred[..., -self.action_dim:]
        z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
        zv_flat = z_visual.reshape((bs, -1))
        zp_flat = z_proprio.reshape((bs, -1))
        concat = torch.cat((zv_flat, zp_flat), dim=1)
        
        return concat
