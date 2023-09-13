import numpy as np
from torch import nn
import torch
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """ Slot Attention mechanism.
    """
    def __init__(self, num_slots, iters, slots_dim, featvec_dim, hidden_dim, resolution, eps,
                 learned_slots, bilevel, learned_factors, scale_inv):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = slots_dim ** -0.5
        self.slots_dim = slots_dim
        self.learned_slots = learned_slots
        self.learned_factors = learned_factors
        self.bilevel = bilevel
        self.resolution = resolution
        self.scale_inv = scale_inv

        self.pos_emb = SoftPositionEmbed(slots_dim, self.resolution)

        if self.learned_slots:
            self.slots = nn.Parameter(torch.randn(1, num_slots, slots_dim))
        else:
            self.slots_mu = nn.Parameter(torch.randn(1, 1, slots_dim))
            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slots_dim))
            nn.init.xavier_uniform_(self.slots_logsigma)

        if self.learned_factors:
            self.s_pos = nn.Parameter(2 * torch.rand(1, num_slots, 2) - 1)
            if self.scale_inv:
                self.s_scale = nn.Parameter(0.1 + 0.01 * torch.randn(1, num_slots, 2))

        self.to_q = nn.Linear(slots_dim, slots_dim)
        self.to_k = nn.Linear(featvec_dim, slots_dim)
        self.to_v = nn.Linear(featvec_dim, slots_dim)
        self.norm_slots = nn.LayerNorm(slots_dim)
        self.norm_inputs = nn.LayerNorm(featvec_dim)

        self.fc1_kv = nn.Linear(slots_dim, 2*slots_dim)
        self.fc2_kv = nn.Linear(2*slots_dim, slots_dim)
        self.norm_pre_fc_k = nn.LayerNorm(slots_dim)
        self.norm_pre_fc_v = nn.LayerNorm(slots_dim)

        self.gru = nn.GRUCell(slots_dim, slots_dim)

        self.fc1_slots = nn.Linear(slots_dim, hidden_dim)
        self.fc2_slots = nn.Linear(hidden_dim, slots_dim)
        self.norm_pre_fc_slots = nn.LayerNorm(slots_dim)

    def step(self, i, slots, k, v, s_pos, s_scale, batch_size):
        slots_prev = slots
        slots = self.norm_slots(slots)
        q = self.to_q(slots)

        k = self.pos_emb(k, s_pos, s_scale).view(*k.shape[:2], -1, k.shape[-1])
        k = self.fc2_kv(F.relu(self.fc1_kv(self.norm_pre_fc_k(k))))
        v = self.pos_emb(v, s_pos, s_scale).view(*v.shape[:2], -1, v.shape[-1])
        v = self.fc2_kv(F.relu(self.fc1_kv(self.norm_pre_fc_v(v))))

        dots = torch.einsum('bid,bijd->bij', q, k) * self.scale
        attn_mask = dots.softmax(dim=1) + self.eps
        attn = attn_mask / attn_mask.sum(dim=-1, keepdim=True)
        updates = torch.einsum('bijd,bij->bid', v, attn)

        if i < self.iters:
            slots = self.gru(
                updates.reshape(-1, self.slots_dim),
                slots_prev.reshape(-1, self.slots_dim)
            )
            slots = slots.reshape(batch_size, -1, self.slots_dim)
            slots = slots + self.fc2_slots(F.relu(self.fc1_slots(self.norm_pre_fc_slots(slots))))
        else:
            slots = slots_prev

        return slots, attn_mask

    def forward(self, inputs, num_slots=None, slots_noise=None):
        batch_size = inputs.shape[0]
        num_slots = num_slots if num_slots is not None else self.num_slots
        
        if self.learned_slots:
            slots_init = self.slots.expand(batch_size, -1, -1)
        else:
            mu = self.slots_mu.expand(batch_size, num_slots, -1)
            sigma = self.slots_logsigma.exp().expand(batch_size, num_slots, -1)
            slots_init = mu + sigma * torch.randn(mu.shape, device=inputs.device)
            del mu, sigma
        if slots_noise is not None:
            slots_init = slots_init + slots_noise * torch.randn(slots_init.shape, device=slots_init.device)
        slots = slots_init
        
        inputs = self.norm_inputs(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        k = k.unsqueeze(1).repeat((1, num_slots, 1, 1, 1))
        v = v.unsqueeze(1).repeat((1, num_slots, 1, 1, 1))

        for i in range(self.iters + 1):
            if self.bilevel and i == self.iters - 1:
                slots = slots.detach() + slots_init - slots_init.detach()

            if i == 0:
                if self.learned_factors:
                    s_pos = self.s_pos.expand(batch_size, -1, -1)
                    s_pos = s_pos.clip(-1., 1.).unsqueeze(2).unsqueeze(2)
                    if self.scale_inv:
                        s_scale = self.s_scale.expand(batch_size, -1, -1)
                        s_scale = s_scale.clip(0.001, 2.).unsqueeze(2).unsqueeze(2)
                    else:
                        s_scale = None
                else:
                    s_pos = 2 * torch.rand(batch_size, num_slots, 1, 1, 2, device=inputs.device) - 1
                    if self.scale_inv:
                        s_scale = 0.1 + 0.1 * torch.randn(batch_size, num_slots, 1, 1, 2, device=inputs.device)
                        s_scale = s_scale.clip(0.001, 2.)
                    else:
                        s_scale = None
            slots, attn_mask = self.step(i, slots, k, v, s_pos, s_scale, batch_size)

            grid_flattened = torch.flatten(self.pos_emb.grid, 1, 2)
            s_pos = torch.einsum('bij,bjd->bid', attn_mask, grid_flattened) / attn_mask.sum(dim=-1, keepdim=True)
            grid_flattened = grid_flattened.unsqueeze(1).repeat((1, num_slots, 1, 1))
            if self.scale_inv:
                s_scale = torch.einsum('bij,bijd->bid', attn_mask+1e-8, (grid_flattened-s_pos.unsqueeze(2))**2)
                s_scale = torch.sqrt(s_scale / (attn_mask+1e-8).sum(dim=-1, keepdim=True))
            s_pos = s_pos.view(*s_pos.shape[:2], 1, 1, s_pos.shape[-1])
            if self.scale_inv:
                s_scale = s_scale.view(*s_scale.shape[:2], 1, 1, s_scale.shape[-1])
                s_scale = s_scale.clip(0.001, 2.)
            del grid_flattened

        return slots, s_pos, s_scale, attn_mask


def build_grid(resolution):
    """ Build 2d grid of specified resolution with ranges [-1, 1].
    """
    ranges = [np.linspace(-1., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="xy")
    grid = np.stack(grid, axis=-1)
    grid = np.expand_dims(grid, axis=0)
    return torch.from_numpy(grid.astype(np.float32))


class SoftPositionEmbed(nn.Module):
    def __init__(self, proj_dim, resolution):
        """ Soft positional embedding layer.
        """
        super().__init__()
        self.embedding = nn.Linear(2, proj_dim, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs, positions, scales):
        grid = self.grid - positions
        if scales is not None:
            grid = grid / (scales * 5)
        grid = self.embedding(grid)
        return inputs + grid


class Encoder(nn.Module):
    """ CNN backbone used to encode feature vectors.
    """
    def __init__(self, hid_dim, small_arch):
        super().__init__()
        if not small_arch:
            self.conv1 = nn.Conv2d(3, hid_dim, 5, stride=(2, 2), padding = 2)
            self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, stride=(1, 1), padding = 2)
            self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, stride=(1, 1), padding = 2)
            self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, stride=(1, 1), padding = 2)
        else:
            self.conv1 = nn.Conv2d(3, hid_dim, 5, padding = 2)
            self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
            self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
            self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = x.permute(0,2,3,1).contiguous()
        return x


class Decoder(nn.Module):
    """ ConvTranspose2d layers used to decode an object based on its broadcasted slot.
    """
    def __init__(self, hid_dim, slots_dim, small_arch):
        super().__init__()
        if not small_arch:
            self.conv1 = nn.ConvTranspose2d(slots_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
            self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
            self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
            self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
            self.conv5 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
            self.conv6 = nn.ConvTranspose2d(hid_dim, 4, 3, stride=(1, 1), padding=1)
        else:
            self.conv1 = nn.ConvTranspose2d(slots_dim, hid_dim, 5, stride=(1, 1), padding=2)
            self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
            self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
            self.conv4 = nn.ConvTranspose2d(hid_dim, 4, 3, stride=(1, 1), padding=1)
        self.small_arch = small_arch

    def forward(self, x):
        x = x.permute(0,3,1,2).contiguous()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        if not self.small_arch:
            x = F.relu(x)
            x = self.conv5(x)
            x = F.relu(x)
            x = self.conv6(x)
        x = x.permute(0,2,3,1).contiguous()
        return x


class ISA(nn.Module):
    def __init__(self, resolution, num_slots, num_iterations, slots_dim, encdec_dim, small_arch,
                 learned_slots=True, bilevel=True, learned_factors=True, scale_inv=True):
        """ Invariant Slot Attention autoencoder. Extention of the PyTorch-based Slot Attention implementation
            from https://github.com/evelinehong/slot-attention-pytorch/blob/master/model.py. 
        """
        super().__init__()
        self.slots_dim = slots_dim
        self.encdec_dim = encdec_dim
        self.enc_resolution = resolution if small_arch else (64, 64)
        self.dec_resolution = resolution if small_arch else (16, 16)
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.small_arch = small_arch
        self.scale_inv = scale_inv

        self.enc = Encoder(self.encdec_dim, self.small_arch)
        self.dec = Decoder(self.encdec_dim, self.slots_dim, self.small_arch)

        self.dec_pos_emb = SoftPositionEmbed(self.slots_dim, self.dec_resolution)
        
        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            iters=self.num_iterations,
            slots_dim=self.slots_dim,
            featvec_dim=self.encdec_dim,
            hidden_dim=128,
            resolution=self.enc_resolution,
            eps=1e-8,
            learned_slots=learned_slots,
            bilevel=bilevel,
            learned_factors=learned_factors,
            scale_inv=self.scale_inv)

    def decode(self, slots, s_pos, s_scale):
        # broadcast slots to a 2D grid
        slots_broadcast = slots.unsqueeze(2).unsqueeze(3)
        slots_broadcast = slots_broadcast.repeat((1, 1, *self.dec_resolution, 1))
        # augment broadcasted slots with rel pos emb
        slots_broadcast = self.dec_pos_emb(slots_broadcast, s_pos, s_scale)
        # collapse batch and slot dims
        slots_broadcast = slots_broadcast.view(-1, *slots_broadcast.shape[2:])

        # feed broadcasted slots to decoder
        x = self.dec(slots_broadcast)
        # undo combination of batch and slot dim
        x = x.reshape(slots.shape[0], -1, x.shape[1], x.shape[2], 4)
        textures, masks = x.split([3, 1], dim=-1)
        del x
        # normalize alpha masks over slot dim
        masks = nn.Softmax(dim=1)(masks)

        # combine textures and masks to get final reconstruction
        reconstruction = torch.sum(textures * masks, dim=1)
        reconstruction = reconstruction.permute(0,3,1,2)

        return reconstruction, textures, masks

    def forward(self, image, num_slots=None, slots_noise=None):
        num_slots = self.num_slots if num_slots is None else num_slots

        # encode image into slots
        x = self.enc(image)
        slots, s_pos, s_scale, attn_mask = self.slot_attention(x, num_slots, slots_noise)
        del x

        # decode slots into reconstruction
        reconstruction, textures, masks = self.decode(slots, s_pos, s_scale)

        # concat position and scale factors to slots
        slots = torch.concat([slots, s_pos.squeeze(2).squeeze(2)], dim=-1)
        if self.scale_inv:
            slots = torch.concat([slots, s_scale.squeeze(2).squeeze(2)], dim=-1)

        return reconstruction, textures, masks, slots, attn_mask