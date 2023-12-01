from torch import nn
import torch.nn.functional as F


class PropertyClf(nn.Module):
    def __init__(self, texture_dim=32, shape_dim=32, color_out_dim=6, material_out_dim=None, shape_out_dim=19):
        super().__init__()
        self.texture_dim = texture_dim
        self.color_out_dim = color_out_dim
        self.material_out_dim = material_out_dim
        if color_out_dim is not None:
            self.color_hidden = nn.Linear(texture_dim, 256)
            self.color_out = nn.Linear(256, color_out_dim)
        if material_out_dim is not None:
            self.material_hidden = nn.Linear(texture_dim, 256)
            self.material_out = nn.Linear(256, material_out_dim)
        self.shape_hidden = nn.Linear(shape_dim, 256)
        self.shape_out = nn.Linear(256, shape_out_dim)

    def forward(self, x, inverse):
        batch_size, num_slots = x.shape[0], x.shape[1]
        x = x.view(-1, x.shape[-1])
        x_texture, x_shape = x[:, :self.texture_dim], x[:, self.texture_dim:]

        if inverse:
            x_texture_ = x_shape
            x_shape_ = x_texture
        else:
            x_texture_ = x_texture
            x_shape_ = x_shape

        out = {}
        if self.color_out_dim is not None:
            out["color"] = self.color_out(F.leaky_relu(self.color_hidden(x_texture_))).view(batch_size, num_slots, -1)
        if self.material_out_dim is not None:
            out["material"] = self.material_out(F.leaky_relu(self.material_hidden(x_texture_))).view(batch_size, num_slots, -1)
        out["shape"] = self.shape_out(F.leaky_relu(self.shape_hidden(x_shape_))).view(batch_size, num_slots, -1)
        return out