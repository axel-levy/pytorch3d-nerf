import torch
import torch.nn as nn
from typing import Union
import numpy as np

from pytorch3d.renderer.cameras import PerspectiveCameras


class CNN(nn.Module):
    def __init__(
            self,
            resolution: int,
            depth: int,
            channels: int,
            kernel_size: int,
            out_dim: int
    ) -> None:
        super(CNN, self).__init__()

        nl = nn.ReLU
        cnn = []
        in_channels = 3
        out_channels = channels
        final_size = resolution
        for _ in range(depth):
            cnn.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'))
            in_channels = out_channels
            cnn.append(nl())
            out_channels = 2 * in_channels
            cnn.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'))
            in_channels = out_channels
            cnn.append(nl())
            cnn.append(nn.AvgPool2d(2))
            final_size = final_size // 2
            cnn.append(nn.GroupNorm(channels, in_channels))
        cnn.append(nn.Conv2d(in_channels, out_dim, final_size, padding='valid'))
        self.cnn = nn.Sequential(*cnn)
        self.out_dim = out_dim

    def forward(
            self,
            image: torch.Tensor
    ) -> torch.Tensor:
        """
        image: [batch_size, resolution, resolution, 3]

        output: [batch_size, out_dim]
        """
        batch_size = image.shape[0]
        return self.cnn(image[:, None]).reshape(batch_size, self.out_dim)


def latents_to_azimuth_elevation(latents: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
    """
    latents: [batch_size, 2]

    output: [batch_size], [batch_size]
    """
    azimuth = latents[..., 0]
    elevation = nn.Sigmoid()(latents[..., 0]) * np.pi / 2.0
    return azimuth, elevation


def azimuth_elevation_to_direction(
        azimuth: torch.Tensor,
        elevation: torch.Tensor
) -> torch.Tensor:
    """
    azimuth: [batch_size]
    elevation: [batch_size]

    output: [batch_size, 3]
    """
    z = torch.sin(elevation)
    x = torch.cos(elevation) * torch.cos(azimuth)
    y = torch.cos(elevation) * torch.sin(azimuth)
    return torch.cat([x[..., None], y[..., None], z[..., None]], -1)


def direction_to_rotmat(
        direction: torch.Tensor
) -> torch.Tensor:
    """
    direction: [batch_size, 3]

    output: [batch_size, 3, 3]
    """
    batch_size = direction.shape[0]
    up = torch.tensor([0., 0., 1.], dtype=torch.float32, device=direction.device)[None].repeat(batch_size, 1)
    z = -direction
    left = torch.cross(up, z, dim=-1)
    x = left / torch.norm(left, dim=-1, keepdim=True)
    y = torch.cross(z, x, dim=-1)
    return torch.cat([x[..., None], y[..., None], z[..., None]], -1)


def rotmat_to_direction(
        rotmat: torch.Tensor
) -> torch.Tensor:
    """
    rotmat: [batch_size, 3, 3]

    output: [batch_size, 3]
    """
    unitvec = torch.tensor([0., 0., 1.], dtype=torch.float32, device=rotmat.device)
    return -(rotmat @ unitvec)


def direction_to_azimuth_elevation(
        direction: torch.Tensor
) -> Union[torch.Tensor, torch.Tensor]:
    """
    direction: [batch_size, 3]

    output: [batch_size], [batch_size]
    """
    elevation = torch.asin(direction[..., 2])
    azimuth = torch.atan2(direction[..., 1], direction[..., 0])
    return azimuth, elevation


class AzimuthElevationCameraPredictor(nn.Module):
    def __init__(
            self,
            resolution: int,
            depth: int,
            channels: int,
            kernel_size: int
    ) -> None:
        super(AzimuthElevationCameraPredictor, self).__init__()

        self.cnn = CNN(resolution, depth, channels, kernel_size, out_dim=2)

    def forward(
            self,
            image: torch.Tensor,
            camera_gt: PerspectiveCameras
    ) -> PerspectiveCameras:
        """
        image: [(batch_size,) resolution, resolution, 3]

        output: [batch_size, out_dim]
        """
        if image.dim() == 3:
            image_in = image[None]
        else:
            image_in = image
        azimuth, elevation = latents_to_azimuth_elevation(self.cnn(image_in))
        rotmat = direction_to_rotmat(azimuth_elevation_to_direction(azimuth, elevation))
        camera_pred = camera_gt.clone()
        camera_pred.R = rotmat
        return camera_pred
