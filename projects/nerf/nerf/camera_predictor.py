import torch
import torch.nn as nn
from typing import Union, Optional
import numpy as np
from vit_pytorch import ViT

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

        # nl = nn.ReLU
        nl = nn.SiLU
        cnn = []
        in_channels = 3
        out_channels = channels
        final_size = resolution
        for _ in range(depth):
            cnn.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'))
            in_channels = out_channels
            # cnn.append(nl())
            # out_channels = 2 * in_channels
            # cnn.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'))
            # in_channels = out_channels
            # cnn.append(nl())
            # cnn.append(nn.AvgPool2d(2))
            # final_size = final_size // 2
            # cnn.append(nn.GroupNorm(channels, in_channels))
            cnn.append(nn.GroupNorm(channels, in_channels))
            cnn.append(nl())
            cnn.append(nn.MaxPool2d(2))
            final_size = final_size // 2
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
        return self.cnn(image.permute(0, 3, 1, 2)).reshape(batch_size, self.out_dim)


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


def azimuth_elevation_to_2d_coords(
        azimuth: torch.Tensor,
        elevation: torch.Tensor
) -> torch.Tensor:
    """
    azimuth: [batch_size]
    elevation: [batch_size]

    output: [batch_size, 2]
    """
    r = -elevation + np.pi / 2.0
    x = r * torch.cos(azimuth)
    y = r * torch.sin(azimuth)
    return torch.cat([x[..., None], y[..., None]], -1)


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


def rotmat_to_in_plane(
        rotmat: torch.Tensor
) -> torch.Tensor:
    """
    rotmat: [batch_size, 3, 3]

    output: [batch_size]
    """
    batch_size = rotmat.shape[0]
    x = rotmat[:, :, 0]
    y = rotmat[:, :, 1]
    z = rotmat[:, :, 2]
    up = torch.tensor([0., 0., 1.], dtype=torch.float32, device=rotmat.device)[None].repeat(batch_size, 1)
    x_aligned = torch.cross(up, z, dim=-1)
    x_aligned /= torch.norm(x_aligned, dim=-1, keepdim=True)
    h = torch.sum(x * x_aligned, dim=-1)
    v = torch.sum(y * x_aligned, dim=-1)
    in_plane = torch.atan2(v, h)
    return in_plane


def apply_in_plane(
        rotmat: torch.Tensor,
        in_plane: torch.Tensor
) -> torch.Tensor:
    """
    rotmat: [batch_size, 3, 3]
    in_plane: [batch_size]

    output: [batch_size, 3, 3]
    """
    correction_matrix = torch.cat([
        torch.cos(in_plane)[..., None], -torch.sin(in_plane)[..., None], torch.zeros_like(in_plane)[..., None],
        torch.sin(in_plane)[..., None], torch.cos(in_plane)[..., None], torch.zeros_like(in_plane)[..., None],
        torch.zeros_like(in_plane)[..., None], torch.zeros_like(in_plane)[..., None],
        torch.ones_like(in_plane)[..., None]
    ], -1).reshape(-1, 3, 3)
    return torch.sum(rotmat[..., None] * correction_matrix.permute(0, 2, 1)[:, None], dim=2)


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


def rotmat_to_2d_coords(
        rotmat: torch.Tensor
) -> torch.Tensor:
    """
    rotmat: [batch_size, 3, 3]

    output: [batch_size, 2]
    """
    azimuth, elevation = direction_to_azimuth_elevation(rotmat_to_direction(rotmat))
    return azimuth_elevation_to_2d_coords(azimuth, elevation)


def align(
        rotmat_ref: torch.Tensor,
        rotmat_rot: torch.Tensor,
        index: int
) -> Union[torch.Tensor, float]:
    """
    rotmat_ref: [batch_size, 3, 3]
    rotmat_rot: [batch_size, 3, 3]
    index: int

    output: [3, 3], float
    """
    alignment = rotmat_ref[index] @ rotmat_rot[index].T
    rotmat_aligned = alignment @ rotmat_rot
    norm = torch.sum((rotmat_ref - rotmat_aligned) ** 2)
    return alignment, norm


def directions_to_out_of_planes(
        direction_1: torch.Tensor,
        direction_2: torch.Tensor
) -> torch.Tensor:
    """
    direction_1: [batch_size, 3]
    direction_2: [batch_size, 3]

    output: [batch_size], out-of-plane angles in rad.
    """
    return torch.arccos(torch.clamp(torch.sum(direction_1 * direction_2, -1), min=-1.0, max=1.0))


def align_rotation_set(
        rotmat_ref: torch.Tensor,
        rotmat_rot: torch.Tensor,
        n_max: int = 16
) -> Union[torch.Tensor, torch.Tensor]:
    """
    rotmat_ref: [batch_size, 3, 3]
    rotmat_rot: [batch_size, 3, 3]
    n_max: int

    output: [3, 3], [batch_size, 3, 3] and rotmat_aligned = alignment @ rotmat_rot ~ rotmat_ref
    """
    batch_size = rotmat_ref.shape[0]
    n_iter = np.min([batch_size, n_max])
    best_alignment = None
    min_norm = None
    for i in range(n_iter):
        alignment, norm = align(rotmat_ref, rotmat_rot, i)
        if min_norm is None or min_norm > norm:
            min_norm = norm
            best_alignment = alignment
    rotmat_aligned = best_alignment @ rotmat_rot
    direction_ref = rotmat_to_direction(rotmat_ref)
    direction_aligned = rotmat_to_direction(rotmat_aligned)
    out_of_planes = directions_to_out_of_planes(direction_ref, direction_aligned)
    mse_oop_deg = torch.mean(out_of_planes) * 180. / np.pi
    medse_oop_deg = torch.median(out_of_planes) * 180. / np.pi
    return best_alignment, rotmat_aligned, mse_oop_deg, medse_oop_deg


def align_camera_gt(
        camera_gt: PerspectiveCameras,
        alignment: torch.Tensor
) -> PerspectiveCameras:
    camera_pred = camera_gt.clone()
    rotmat_gt_aligned = alignment.to(camera_pred.R.device).T @ camera_pred.R
    camera_pred.R = rotmat_gt_aligned
    return camera_pred


def latents_to_direction(
        latents: torch.Tensor
) -> torch.Tensor:
    """
    latents: [batch_size, 3]

    output: [batch_size, 3]
    """
    return latents / torch.norm(latents, dim=-1, keepdim=True)


def fix_elevation(
        direction: torch.Tensor,
        fixed_elevation: float
) -> torch.Tensor:
    """
    direction: [batch_size, 3]
    fixed_elevation: float

    output: [batch_size, 3]
    """
    azimuth, _ = direction_to_azimuth_elevation(direction)
    elevation = torch.ones_like(azimuth) * fixed_elevation
    return azimuth_elevation_to_direction(azimuth, elevation)


def direction_to_camera(
        direction: torch.Tensor,
        camera_gt: PerspectiveCameras,
        replication_loss: bool,
        replication_order: int,
        northern_hemisphere: bool,
        no_elevation: bool = False,
        use_gt_in_planes: bool = False
) -> PerspectiveCameras:
    if northern_hemisphere:
        direction = constrain_north(direction)
    if no_elevation:
        # fixed_elevation = np.arctan(0.5 / 4.0)
        fixed_elevation = 0.0
        direction = fix_elevation(direction, fixed_elevation)
    if replication_loss:
        direction_replicated = replicate_direction(direction, replication_order)
        rotmat = direction_to_rotmat(direction_replicated)
        if use_gt_in_planes:
            in_plane = rotmat_to_in_plane(camera_gt.R)
            in_plane = torch.cat([in_plane] * replication_order, dim=0)
            rotmat = apply_in_plane(rotmat, in_plane)
        camera_pred = PerspectiveCameras(
            focal_length=torch.cat([camera_gt.focal_length] * replication_order, dim=0),
            principal_point=torch.cat([camera_gt.principal_point] * replication_order, dim=0),
            R=rotmat,
            T=torch.cat([camera_gt.T] * replication_order, dim=0),
        ).to(camera_gt.device)
    else:
        rotmat = direction_to_rotmat(direction)
        camera_pred = camera_gt.clone()
        if use_gt_in_planes:
            in_plane = rotmat_to_in_plane(camera_gt.R)
            rotmat = apply_in_plane(rotmat, in_plane)
        camera_pred.R = rotmat
    return camera_pred


class PEAzimuthElevationCameraPredictor(nn.Module):
    def __init__(
            self,
            n_images: int,
            replication_loss: bool = False,
            replication_order: int = 2,
            northern_hemisphere: bool = False
    ) -> None:
        super(PEAzimuthElevationCameraPredictor, self).__init__()

        direction_init = torch.tensor(np.random.randn(n_images, 3)).float()
        direction_init /= torch.norm(direction_init, dim=-1, keepdim=True)
        self.table_direction = nn.Parameter(direction_init, requires_grad=True)
        self.replication_loss = replication_loss
        self.replication_order = replication_order
        self.northern_hemisphere = northern_hemisphere

    def forward(
            self,
            idx: torch.Tensor,
            camera_gt: PerspectiveCameras
    ) -> PerspectiveCameras:
        direction = latents_to_direction(self.table_direction[idx][None])
        return direction_to_camera(
            direction, camera_gt, self.replication_loss, self.replication_order, self.northern_hemisphere
        )


class AzimuthElevationCameraPredictor(nn.Module):
    def __init__(
            self,
            resolution: int,
            depth: int,
            channels: int,
            kernel_size: int,
            replication_loss: bool = False,
            replication_order: int = 2,
            northern_hemisphere: bool = False,
            no_elevation: bool = False,
            n_noisy_epochs: int = 0,
            use_gt_in_planes: bool = False,
            transformer: bool = False
    ) -> None:
        super(AzimuthElevationCameraPredictor, self).__init__()

        self.transformer = transformer
        if not transformer:
            self.cnn = CNN(resolution, depth, channels, kernel_size, out_dim=3)
        else:
            self.vit = ViT(
                image_size=resolution,
                patch_size=resolution // 16,
                num_classes=3,
                dim=256,
                depth=6,
                heads=16,
                mlp_dim=256
            )
        self.replication_loss = replication_loss
        self.replication_order = replication_order
        self.northern_hemisphere = northern_hemisphere
        self.no_elevation = no_elevation
        self.n_noisy_epochs = n_noisy_epochs
        self.use_gt_in_planes = use_gt_in_planes

    def forward(
            self,
            image: torch.Tensor,
            camera_gt: PerspectiveCameras,
            epoch: Optional[int] = None
    ) -> PerspectiveCameras:
        if epoch is not None and epoch < self.n_noisy_epochs:
            batch_size = image.shape[0]
            direction = latents_to_direction(
                torch.tensor(np.random.randn(batch_size, 3)).float().to(image.device))
        else:
            if not self.transformer:
                direction = latents_to_direction(self.cnn(image))
            else:
                direction = latents_to_direction(self.vit(image.permute(0, 3, 1, 2)))
        return direction_to_camera(
            direction, camera_gt, self.replication_loss, self.replication_order, self.northern_hemisphere,
            no_elevation=self.no_elevation, use_gt_in_planes=self.use_gt_in_planes
        )


def constrain_north(
        direction: torch.Tensor
) -> torch.Tensor:
    """
    direction: [batch_size, 3]

    output: [batch_size, 3]
    """
    direction[..., 2][direction[..., 2] < 0] = -direction[..., 2][direction[..., 2] < 0]
    return direction


def replicate_direction(
        direction: torch.Tensor,
        replication_order: int
) -> torch.Tensor:
    """
    direction: [batch_size, 3]
    replication_order: int

    output: [(replication_order * ) batch_size, 3]
    """
    azimuth, elevation = direction_to_azimuth_elevation(direction)
    elevation_replicated = torch.cat([elevation] * replication_order, dim=0)
    azimuth_replicated = torch.cat([
        azimuth + angle for angle in np.arange(replication_order) * 2.0 * np.pi / replication_order
    ], dim=0)
    direction_replicated = azimuth_elevation_to_direction(azimuth_replicated, elevation_replicated)
    return direction_replicated


def select_paths(
        x: torch.Tensor,
        activated_paths: torch.Tensor,
        replication_order: int
) -> torch.Tensor:
    """
    x: [replication_order * batch_size, ...]
    activated_paths: [batch_szie]
    replication_order: int

    output: [replication_order, batch_size, ...]
    """
    return torch.cat(
        [t[None] for t in torch.chunk(x, replication_order)], 0
    )[activated_paths, np.arange(len(activated_paths))]
