# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def calc_mse(x: torch.Tensor, y: torch.Tensor):
    """
    Calculates the mean square error between tensors `x` and `y`.
    """
    return torch.mean((x - y) ** 2)


def calc_psnr(x: torch.Tensor, y: torch.Tensor):
    """
    Calculates the Peak-signal-to-noise ratio between tensors `x` and `y`.
    """
    mse = calc_mse(x, y)
    psnr = -10.0 * torch.log10(mse)
    return psnr


def calc_replication_loss(x: torch.Tensor, y: torch.Tensor, replication_order: int):
    """
    x: [(replication_order * ) batch_size, ...]
    y: [(replication_order * ) batch_size, ...]
    """
    replication_order_batch_size = x.shape[0]
    batch_size = replication_order_batch_size // replication_order
    y = y.reshape(replication_order, batch_size, -1)
    x = x.reshape(replication_order, batch_size, -1)
    square_errors = torch.mean((x - y) ** 2, dim=-1)
    min_distances, activated_paths = torch.min(square_errors, 0)
    return min_distances.mean(), activated_paths


def mix_paths(x: torch.Tensor, replication_order: int):
    """
    x: [(replication_order * ) batch_size, ...]
    """
    replication_order_batch_size = x.shape[0]
    batch_size = replication_order_batch_size // replication_order
    x = x.reshape(replication_order, batch_size, *x.shape[1:])
    x = torch.cat([torch.mean(x, dim=1)[None] * replication_order], 0)
    x = x.reshape(-1, *x.shape[2:])
    return x


def sample_images_at_mc_locs(
    target_images: torch.Tensor,
    sampled_rays_xy: torch.Tensor,
):
    """
    Given a set of pixel locations `sampled_rays_xy` this method samples the tensor
    `target_images` at the respective 2D locations.

    This function is used in order to extract the colors from ground truth images
    that correspond to the colors rendered using a Monte Carlo rendering.

    Args:
        target_images: A tensor of shape `(batch_size, ..., 3)`.
        sampled_rays_xy: A tensor of shape `(batch_size, S_1, ..., S_N, 2)`.

    Returns:
        images_sampled: A tensor of shape `(batch_size, S_1, ..., S_N, 3)`
            containing `target_images` sampled at `sampled_rays_xy`.
    """
    ba = target_images.shape[0]
    dim = target_images.shape[-1]
    spatial_size = sampled_rays_xy.shape[1:-1]

    # The coordinate grid convention for grid_sample has both x and y
    # directions inverted.
    xy_sample = -sampled_rays_xy.view(ba, -1, 1, 2).clone()
    inputs = target_images.permute(0, 3, 1, 2)

    images_sampled = torch.nn.functional.grid_sample(
        inputs,
        xy_sample,
        align_corners=True,
        mode="bilinear",
    )
    return images_sampled.permute(0, 2, 3, 1).view(ba, *spatial_size, dim)
