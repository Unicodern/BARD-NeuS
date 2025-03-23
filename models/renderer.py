import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
import matplotlib.pyplot as plt
from icecream import ic


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64  # resolution = 64
    # 获取提取范围立方体在x、y、z轴上的范围（min~max），并且按resolution为总步长
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    # 初始化立方体的sdf值
    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    # 网格
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)  # [N, N, N]
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans 防止权重为0
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # [batch_size, n_samples - 1]
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        # 相邻两个采样点之间的距离
        dists = z_vals[..., 1:] - z_vals[..., :-1]  # [batch_size, n_samples - 1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)  # [batch_size, n_samples]
        mid_z_vals = z_vals + dists * 0.5  # 相邻两个采样点的中点，以中点为新采样点

        # Section midpoints
        # 中点在世界坐标系下的位置
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # [batch_size, n_samples, 3]

        # 原点到中点的距离，clip让小于1的值始终为1（单位球内部），保留大于1的值（单位球外部）
        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)  # keepdim = True --> [batch_size, n_samples, 1]
        # 归一化处理，球体内的点相当于没有作处理，还增加了一倍值为1
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # [batch_size, n_samples, 4]

        # 单位方向向量
        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)  # [batch_size, n_samples, 3]

        # reshape(-1, 4)
        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))  # [batch_size * n_samples, 4]
        dirs = dirs.reshape(-1, 3)  # [batch_size * n_samples, 3]

        # 通过nerf网络获取背景的概率密度(?)和rgb值
        density, sampled_color = nerf(pts, dirs)
        # rgb归一化
        sampled_color = torch.sigmoid(sampled_color)  # [batch_size * n_samples, 3]
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)  # [batch_size, n_samples]
        alpha = alpha.reshape(batch_size, n_samples)  # [batch_size, n_samples]
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]  # [batch_size, n_samples]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)  # [batch_size, n_samples, 3]
        # 背景颜色 = sum(权重 * 采样点颜色)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)  # [batch_size, 1, 3]
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,  # [batch_size, 1, 3]
            'sampled_color': sampled_color,  # [batch_size, n_samples, 3]
            'alpha': alpha,  # [batch_size, n_samples]
            'weights': weights,  # [batch_size, n_samples]
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        # 世界坐标系下粗采样点的位置
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # [batch_size, n_samples, 3]
        # 坐标原点到世界坐标系下粗采样点的距离
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)  # [batch_size, n_samples]
        # 采样点每相邻两点为一个采样段
        # 采样段两端点任一端在半径为的单位球内，则采样段有效
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)  # [batch_size, n_samples - 1]

        sdf = sdf.reshape(batch_size, n_samples)  # [batch_size, n_samples]
        # 世界坐标系下近点集和远点集的sdf
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]  # [batch_size, n_samples]
        # 近点集和远点集
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]  # [batch_size, n_samples - 1]
        # 粗采样点sdf的均值
        mid_sdf = (prev_sdf + next_sdf) * 0.5  # [batch_size, n_samples - 1]
        # 两个采样点的sdf值和采样点间距之比
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)  # [batch_size, n_samples - 1]

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)  # [batch_size, n_samples - 1]
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)  # [batch_size, n_samples - 1, 2]
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)  # 选择最小值  [batch_size, n_samples - 1]
        # clip函数用于将cos_val限定到区间[-1000， 0]，并且保证是有效采样段i m
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere  # [batch_size, n_samples - 1]

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5  # prev_sdf [batch_size, n_samples - 1]
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5  # next_sdf [batch_size, n_samples - 1]
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)  # [batch_size, n_samples - 1]
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)  # [batch_size, n_samples - 1]
        # 当采样段无效时，alpha为0，当采样段有效时，alpha大于0
        # 因此alpha从0变为大于0，则表示采样段穿入单位球体，反之则是穿出球体
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)  # [batch_size, n_samples - 1]
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]  # [batch_size, n_samples - 1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    # 确定精采样点后，由SDFNetwork计算对应的sdf值
    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape

        # Section length
        # 相邻两个采样点的距离
        dists = z_vals[..., 1:] - z_vals[..., :-1]  # [batch_size, n_samples - 1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)  # [batch_size, n_samples]
        # 取每两个采样点的中点，作为新的采样点
        mid_z_vals = z_vals + dists * 0.5  # [batch_size, n_samples]

        # Section midpoints
        # 中点（即新采样点）在世界坐标系下的位置
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # [batch_size, n_samples, 3]
        # 单位方向向量
        dirs = rays_d[:, None, :].expand(pts.shape)  # [batch_size, n_samples, 3]

        pts = pts.reshape(-1, 3)  # [batch_size * n_samples, 3]
        dirs = dirs.reshape(-1, 3)  # [batch_size * n_samples, 3]

        # sdf network的输出
        sdf_nn_output = sdf_network(pts)  # [batch_size * n_samples, 257]
        # sdf值
        sdf = sdf_nn_output[:, :1]  # [batch_size * n_samples, 1]
        # 256维的隐藏特征
        feature_vector = sdf_nn_output[:, 1:]  # [batch_size * n_samples, 256]

        # 梯度信息
        gradients = sdf_network.gradient(pts).squeeze()  # 将所有一维数据压缩 [batch_size * n_samples, 1, 3] --> [batch_size * n_samples, 3]
        # 采样点的颜色，reshape前的shape：[batch_size * n_samples, 3]
        sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)  # [batch_size, n_samples, 3]

        # 单参数网络  可训练参数网络
        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # [1, 1]
        inv_s = inv_s.expand(batch_size * n_samples, 1)  # [batch_size * n_samples, 1] 所有值都是上一行计算出的那一个值

        # 单位方向向量乘上梯度信息
        true_cos = (dirs * gradients).sum(-1, keepdim=True)  # [batch_size * n_samples, 3] --> [batch_size * n_samples, 1]

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive [batch_size * n_samples, 1]  iter_cos <= 0

        # Estimate signed distances at section points
        # 估计采样点sdf的取值区间
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5  # [batch_size * n_samples, 1]
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5  # [batch_size * n_samples, 1]

        # 归一化，分别时采样点sdf区段的始端和末端
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)  # [batch_size * n_samples, 1]
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)  # [batch_size * n_samples, 1]

        # 采样sdf区段的长度
        p = prev_cdf - next_cdf  # [batch_size * n_samples, 1]
        # 起始端
        c = prev_cdf  # [batch_size * n_samples, 1]

        # 长度/起始端，将范围限制在0~1
        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)  # [batch_size, n_samples]

        # 采样点的长度
        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)  # [batch_size, n_samples]
        # 单位球的内部
        inside_sphere = (pts_norm < 1.0).float().detach()  # [batch_size, n_samples]
        # 球体的范围从1放宽到1.2
        relax_inside_sphere = (pts_norm < 1.2).float().detach()  # [batch_size, n_samples]

        # Render with background
        # womask.conf
        # background_alpha、background_sampled_color分别是render_core_outside()中通过nerf网络计算出的关于背景的alpha值和采样点颜色
        if background_alpha is not None:
            # aplha分两部分，（1）保留下来的位于球体内部的部分；（2）球体外的部分则用背景采样点的alpha值替换
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)  # [batch_size, n_samples + n_outside]
            # 采样点颜色类似alpha，分球体内和球体外两部分
            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]  # [batch_size, n_samples, 3]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)  # [batch_size, n_samples + n_outside, 3]

        # 对应原文公式Ti的计算方式
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]  # [batch_size, n_samples + n_outside]
        weights_sum = weights.sum(dim=-1, keepdim=True)  # [batch_size, 1]

        # 对应原文公式11
        color = (sampled_color * weights[:, :, None]).sum(dim=1)  # [batch_size, 1, 3] --> [batch_size, 3]
        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # Eikonal loss
        # 梯度的二范数的平方
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        # 保留在放宽限制的球体内部的梯度
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        # --------------------------预测深度----------------------------
        depth_pred = (mid_z_vals * weights[:, :n_samples]).sum(dim=1, keepdim=True)  # [batch_size, 1]
        # depth_pred = torch.sum(weights * mid_z_vals, 1, keepdims=True) / (weights.sum(dim=1, keepdims=True) +1e-8)
        # depth_variance = ((mid_z_vals - depth) ** 2 * weights[:, :n_samples]).sum(dim=-1, keepdim=True)
        pts_pred = rays_o + depth_pred * rays_d  # [batch_size, 3]
        # ------------------------------------------------------------

        return {
            'color': color,  # 前景颜色 [batch_size, 3]
            'sdf': sdf,  # sdf值 [batch_size * n_samples, 1]
            'dists': dists,  # 相邻两个采样点的距离
            'gradients': gradients.reshape(batch_size, n_samples, 3),  # 梯度 [batch_size, n_samples, 3]
            's_val': 1.0 / inv_s,  # 可训练参数 [batch_size * n_samples, 1]
            'mid_z_vals': mid_z_vals,  # 采样点（中点）位置 [batch_size, n_samples]
            'weights': weights,  # 采样点（中点）权重 [batch_size, n_samples + n_outside]
            'cdf': c.reshape(batch_size, n_samples),  # 采样点（中点）sdf起始端值 [batch_size, n_samples]
            'gradient_error': gradient_error,  # 梯度信息，位于放宽限制的球体内的采样点梯度均保留 只有一个值 []
            'inside_sphere': inside_sphere,  # 单位球的内部 [batch_size, n_samples]
            'depth_pred': depth_pred,  # 深度预测值 [batch_size, 1]
            'pts_pred': pts_pred  # 世界坐标预测值 [batch_size, 3]
        }

    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)  # 生成0~1之间总共n_samples个等距的值
        z_vals = near + (far - near) * z_vals[None, :]  # 粗采样点点集 [batch_size,n_samples]

        # rays背景采样，在无mask分割的情况下才会进行背景采样
        z_vals_outside = None
        if self.n_outside > 0:
            # 同z_vals，也是均匀采样
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb  # 扰动系数

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            # 从-0.5~0.5的均匀分布范围内为每个ray的所有粗采样点（总共batch_size个）随机选取一个扰动系数
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            # 原来的粗采样点时均匀采样的，现在对其进行扰动
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])  # 背景采样点区间中，第一个点和最后一个点的中点
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)  # 远点点集
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)  # 近点点集
                # 对每条ray的每个背景采样点都单独设置一个扰动，总共有batch_size * n_outside个扰动系数
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])  # 扰动系数
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand  # 添加扰动

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                # 对应原文公式：p(t) = {o + tv | t >= 0}
                # 即采样点在世界坐标系下的位置 = 光心 + 单位方向向量 * 采样点
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                # 获取sdf值，即值为0的等势面
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)  # [batch_size,n_samples,1]

                # 粗采样点细分精采样点不断逼近sdf=0的等势面
                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        # Background model
        if self.n_outside > 0:
            # 前景采用点 + 背景采样点
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)  # 背景

            background_sampled_color = ret_outside['sampled_color']  # 采样点颜色
            background_alpha = ret_outside['alpha']  # 采样点的...（权重因子？）

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio)

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        return {
            'color_fine': color_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'z_vals': ret_fine['mid_z_vals'],
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere'],
            'depth_pred': ret_fine['depth_pred'],
            'pts_pred': ret_fine['pts_pred']
        }

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
