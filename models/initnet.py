import torch
import torch.nn as nn
from einops import rearrange
from .vqvae import ResBlock

class InitNetEncoder(nn.Module):
    def __init__(self, in_channel, channel, out_channel, n_res_channel, stride):
        super(InitNetEncoder, self).__init__()

        blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]
        blocks.append(ResBlock(channel, n_res_channel))
        blocks.extend([
                nn.Conv2d(channel, channel // 2, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ])
        blocks.append(ResBlock(channel, n_res_channel))
        blocks.extend([
            nn.Conv2d(channel, out_channel, 3, padding=1),
        ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class InitNet(nn.Module):
    def __init__(
        self,
        in_channel,
        num_actions,
        action_dim,
        encoder_dim=1024, 
        channel=128,
        n_res_channel=32
    ):
        super(InitNet, self).__init__()
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.encoder = InitNetEncoder(in_channel, channel, encoder_dim, n_res_channel, 4)
        self.act_proj = nn.Linear(encoder_dim, num_actions * action_dim)
       

    def forward(self, cur_state, goal_state):
        '''
            cur_state: (b, num_patches, emb_dim)
            goal_state: (b, num_patches, emb_dim)
        '''
        input = torch.cat((cur_state, goal_state), dim=2)
        num_patches = cur_state.shape[1]
        num_side_patches = int(num_patches ** 0.5)
        input = rearrange(input, "b (h w) d -> b d h w", h=num_side_patches, w=num_side_patches)
        x = self.encoder(input)
        x = torch.flatten(x, start_dim=1)
        x = self.act_proj(x)
        x = rearrange(x, "b (t d) -> b t d", t=self.num_actions)
        return x
