import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder

class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones([1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, dim]))

    def forward(self, x):
        return x * self.alpha + self.beta

class Feedforward(nn.Module):
    def __init__(self,
                 dim,
                 multires=0,
                 geometric_init=True,
                 weight_norm=True):
        super().__init__()

        self.lin = nn.Linear(dim, dim)
        self.soft_plus = nn.Softplus(beta=100)

        if geometric_init and multires > 0:
            torch.nn.init.constant_(self.lin.bias, 0.0)
            torch.nn.init.normal_(self.lin.weight, 0.0, np.sqrt(2) / np.sqrt(dim))

        if weight_norm:
            self.lin = nn.utils.weight_norm(self.lin)

        self.network = nn.Sequential(
            self.lin,
            self.soft_plus,
        )

    def forward(self, x):
        return self.network(x)


class ResBlock(nn.Module):
    def __init__(self,
                 dim,
                 multires=0,
                 init_values=1e-4,
                 geometric_init=True,
                 weight_norm=True):
        super().__init__()
        self.pre_affine = Affine(dim)
        self.post_affine = Affine(dim)

        self.lin = nn.Linear(dim, dim)
        if geometric_init and multires > 0:
            torch.nn.init.constant_(self.lin.bias, 0.0)
            torch.nn.init.normal_(self.lin.weight, 0.0, np.sqrt(2) / np.sqrt(dim))

        if weight_norm:
            self.lin = nn.utils.weight_norm(self.lin)

        self.soft_plus = nn.Softplus(beta=100)
        self.feedforward = nn.Sequential(
            Feedforward(dim=dim,
                        multires=multires,
                        geometric_init=geometric_init,
                        weight_norm=weight_norm))
        self.layer_scale_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        res_1 = self.lin(self.pre_affine(x))
        res_1 = self.soft_plus(res_1)
        x = x + self.layer_scale_1 * res_1
        res_2 = self.feedforward(self.post_affine(x))
        x = x + self.layer_scale_2 * res_2
        return x


class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_blocks,
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        self.dim_in = d_in
        self.dim_out = d_out
        self.dim_hidden = d_hidden
        self.scale = scale

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            self.dim_in = input_ch

        self.head_lin = nn.Linear(self.dim_in, self.dim_hidden)
        self.skip_lin = nn.Linear(self.dim_hidden, self.dim_hidden - self.dim_in)
        self.next_lin = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.tail_lin = nn.Linear(self.dim_hidden, self.dim_out)

        if geometric_init and multires > 0:
            torch.nn.init.constant_(self.head_lin.bias, 0.0)
            torch.nn.init.constant_(self.head_lin.weight[:, 3:], 0.0)
            torch.nn.init.normal_(self.head_lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(self.dim_hidden))

            torch.nn.init.constant_(self.skip_lin.bias, 0.0)
            torch.nn.init.normal_(self.skip_lin.weight, 0.0, np.sqrt(2) / np.sqrt(self.dim_hidden))

            torch.nn.init.constant_(self.next_lin.bias, 0.0)
            torch.nn.init.normal_(self.next_lin.weight, 0.0, np.sqrt(2) / np.sqrt(self.dim_hidden - self.dim_in))
            torch.nn.init.constant_(self.next_lin.weight[:, -(self.dim_in - 3):], 0.0)

            if not inside_outside:
                torch.nn.init.normal_(self.tail_lin.weight, mean=np.sqrt(np.pi) / np.sqrt(self.dim_hidden), std=0.0001)
                torch.nn.init.constant_(self.tail_lin.bias, -bias)
            else:
                torch.nn.init.normal_(self.tail_lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(self.dim_hidden), std=0.0001)
                torch.nn.init.constant_(self.tail_lin.bias, bias)

        if weight_norm:
            self.head_lin = nn.utils.weight_norm(self.head_lin)
            self.skip_lin = nn.utils.weight_norm(self.skip_lin)
            self.tail_lin = nn.utils.weight_norm(self.tail_lin)

        self.first_block = ResBlock(dim=self.dim_hidden,
                                    multires=multires,
                                    geometric_init=geometric_init,
                                    weight_norm=weight_norm)
        self.blocks = nn.ModuleList([])
        for _ in range(0, n_blocks - 1):
            self.blocks.append(ResBlock(dim=self.dim_hidden,
                                        multires=multires,
                                        geometric_init=geometric_init,
                                        weight_norm=weight_norm))
        self.soft_plus = nn.Softplus(beta=100)
        self.affine = Affine(self.dim_hidden)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        x = self.head_lin(x)
        x = self.soft_plus(x)
        x = self.first_block(x)

        x = self.skip_lin(x)

        x = self.soft_plus(x)
        x = torch.cat([x, inputs], 1) / np.sqrt(2)
        x = self.next_lin(x)
        x = self.soft_plus(x)

        for block in self.blocks:
            x = block(x)
        x = self.affine(x)
        x = self.tail_lin(x)

        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x) 
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output, 
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1) 

class ColorResBlock(nn.Module):
    def __init__(self,
                 dims,
                 weight_norm=True,
                 levels=3,
                 init_values=1e-4):
        super().__init__()

        self.pre_affine = Affine(dims)
        self.post_affine = Affine(dims)

        self.lins = [nn.Linear(dims, dims) for _ in range(levels)]
        if weight_norm:
            for i in range(len(self.lins)):
                self.lins[i] = nn.utils.weight_norm(self.lins[i])
        self.relu = nn.ReLU()

        self.lins_len = len(self.lins)

        self.layer_scale_1 = nn.Parameter(init_values * torch.ones((dims)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_values * torch.ones((dims)), requires_grad=True)

    def forward(self, x):
        res_1 = self.lins[0](self.pre_affine(x))
        res_1 = self.relu(res_1)
        x = x + self.layer_scale_1 * res_1

        res_2 = self.post_affine(x)
        for i in range(1, self.lins_len):
            res_2 = self.lins[i](res_2)
            res_2 = self.relu(res_2)

        x = x + self.layer_scale_2 * res_2
        return x

class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.squeeze_out = squeeze_out
        self.dims_in = d_in + d_feature
        self.dims_hidden = d_hidden
        self.dims_out = d_out

        # Positional encoding embedding
        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            self.dims_in += (input_ch - 3)

        self.head_lin = nn.Linear(self.dims_in, self.dims_hidden)
        self.tail_lin = nn.Linear(self.dims_hidden, self.dims_out)

        if weight_norm:
            self.head_lin = nn.utils.weight_norm(self.head_lin)
            self.tail_lin = nn.utils.weight_norm(self.tail_lin)

        self.res_block = ColorResBlock(dims=self.dims_hidden,
                                       weight_norm=weight_norm,
                                       levels=n_layers-2)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors): 
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None
        rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        x = rendering_input

        x = self.head_lin(x)
        x = self.relu(x)

        x = self.res_block(x)

        x = self.tail_lin(x)
        x = self.sigmoid(x)
        return x


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)

