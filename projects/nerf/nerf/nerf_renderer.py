# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch3d.renderer import ImplicitRenderer, ray_bundle_to_ray_points
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene
from visdom import Visdom
from torch.utils.tensorboard import SummaryWriter

from .implicit_function import NeuralRadianceField
from .raymarcher import EmissionAbsorptionNeRFRaymarcher
from .raysampler import NeRFRaysampler, ProbabilisticRaysampler
from .utils import calc_mse, calc_psnr, sample_images_at_mc_locs, calc_replication_loss
from .camera_predictor import AzimuthElevationCameraPredictor, rotmat_to_2d_coords, align_rotation_set, select_paths, align_camera_gt, PEAzimuthElevationCameraPredictor


class RadianceFieldRenderer(torch.nn.Module):
    """
    Implements a renderer of a Neural Radiance Field.

    This class holds pointers to the fine and coarse renderer objects, which are
    instances of `pytorch3d.renderer.ImplicitRenderer`, and pointers to the
    neural networks representing the fine and coarse Neural Radiance Fields,
    which are instances of `NeuralRadianceField`.

    The rendering forward pass proceeds as follows:
        1) For a given input camera, rendering rays are generated with the
            `NeRFRaysampler` object of `self._renderer['coarse']`.
            In the training mode (`self.training==True`), the rays are a set
                of `n_rays_per_image` random 2D locations of the image grid.
            In the evaluation mode (`self.training==False`), the rays correspond
                to the full image grid. The rays are further split to
                `chunk_size_test`-sized chunks to prevent out-of-memory errors.
        2) For each ray point, the coarse `NeuralRadianceField` MLP is evaluated.
            The pointer to this MLP is stored in `self._implicit_function['coarse']`
        3) The coarse radiance field is rendered with the
            `EmissionAbsorptionNeRFRaymarcher` object of `self._renderer['coarse']`.
        4) The coarse raymarcher outputs a probability distribution that guides
            the importance raysampling of the fine rendering pass. The
            `ProbabilisticRaysampler` stored in `self._renderer['fine'].raysampler`
            implements the importance ray-sampling.
        5) Similar to 2) the fine MLP in `self._implicit_function['fine']`
            labels the ray points with occupancies and colors.
        6) self._renderer['fine'].raymarcher` generates the final fine render.
        7) The fine and coarse renders are compared to the ground truth input image
            with PSNR and MSE metrics.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        n_pts_per_ray: int,
        n_pts_per_ray_fine: int,
        n_rays_per_image: int,
        min_depth: float,
        max_depth: float,
        stratified: bool,
        stratified_test: bool,
        chunk_size_test: int,
        n_harmonic_functions_xyz: int = 6,
        n_harmonic_functions_dir: int = 4,
        n_hidden_neurons_xyz: int = 256,
        n_hidden_neurons_dir: int = 128,
        n_layers_xyz: int = 8,
        camera_predictor_type: str = 'gt',
        depth_camera_predictor: int = 4,
        channels_camera_predictor: int = 16,
        kernel_size_camera_predictor: int = 3,
        northern_hemisphere: bool = False,
        n_images: int = 100,
        append_xyz: Tuple[int, ...] = (5,),
        density_noise_std: float = 0.0,
        visualization: bool = False,
        view_dependency: bool = True,
        mask_loss: bool = False,
        replication_loss: bool = False,
        replication_order: int = 2
    ):
        """
        Args:
            image_size: The size of the rendered image (`[height, width]`).
            n_pts_per_ray: The number of points sampled along each ray for the
                coarse rendering pass.
            n_pts_per_ray_fine: The number of points sampled along each ray for the
                fine rendering pass.
            n_rays_per_image: Number of Monte Carlo ray samples when training
                (`self.training==True`).
            min_depth: The minimum depth of a sampled ray-point for the coarse rendering.
            max_depth: The maximum depth of a sampled ray-point for the coarse rendering.
            stratified: If `True`, stratifies (=randomly offsets) the depths
                of each ray point during training (`self.training==True`).
            stratified_test: If `True`, stratifies (=randomly offsets) the depths
                of each ray point during evaluation (`self.training==False`).
            chunk_size_test: The number of rays in each chunk of image rays.
                Active only when `self.training==True`.
            n_harmonic_functions_xyz: The number of harmonic functions
                used to form the harmonic embedding of 3D point locations.
            n_harmonic_functions_dir: The number of harmonic functions
                used to form the harmonic embedding of the ray directions.
            n_hidden_neurons_xyz: The number of hidden units in the
                fully connected layers of the MLP that accepts the 3D point
                locations and outputs the occupancy field with the intermediate
                features.
            n_hidden_neurons_dir: The number of hidden units in the
                fully connected layers of the MLP that accepts the intermediate
                features and ray directions and outputs the radiance field
                (per-point colors).
            n_layers_xyz: The number of layers of the MLP that outputs the
                occupancy field.
            camera_predictor_type: Type of camera predictor.
            depth_camera_predictor: Depth of camera predictor.
            channels_camera_predictor: Number of channels of camera predictor.
            kernel_size_camera_predictor: Kernel size of camera predictor.
            northern_hemisphere: Whether to constrain the directions to the Northern hemisphere.
            n_images: Number of images in the training dataset.
            append_xyz: The list of indices of the skip layers of the occupancy MLP.
                Prior to evaluating the skip layers, the tensor which was input to MLP
                is appended to the skip layer input.
            density_noise_std: The standard deviation of the random normal noise
                added to the output of the occupancy MLP.
                Active only when `self.training==True`.
            visualization: whether to store extra output for visualization.
            view_dependency: Whether the radiance field should be view-dependent.
            mask_loss: Whether to compute the mask_loss.
            replication_loss: Whether to use the replication loss.
            replication_order: Replication order for the replication loss.
        """

        super().__init__()

        # The renderers and implicit functions are stored under the fine/coarse
        # keys in ModuleDict PyTorch modules.
        self._renderer = torch.nn.ModuleDict()
        self._implicit_function = torch.nn.ModuleDict()

        # Init the EA raymarcher used by both passes.
        raymarcher = EmissionAbsorptionNeRFRaymarcher()

        # Parse out image dimensions.
        image_height, image_width = image_size

        for render_pass in ("coarse", "fine"):
            if render_pass == "coarse":
                # Initialize the coarse raysampler.
                raysampler = NeRFRaysampler(
                    n_pts_per_ray=n_pts_per_ray,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    stratified=stratified,
                    stratified_test=stratified_test,
                    n_rays_per_image=n_rays_per_image,
                    image_height=image_height,
                    image_width=image_width,
                )
            elif render_pass == "fine":
                # Initialize the fine raysampler.
                raysampler = ProbabilisticRaysampler(
                    n_pts_per_ray=n_pts_per_ray_fine,
                    stratified=stratified,
                    stratified_test=stratified_test,
                )
            else:
                raise ValueError(f"No such rendering pass {render_pass}")

            # Initialize the fine/coarse renderer.
            self._renderer[render_pass] = ImplicitRenderer(
                raysampler=raysampler,
                raymarcher=raymarcher,
            )

            # Instantiate the fine/coarse NeuralRadianceField module.
            self._implicit_function[render_pass] = NeuralRadianceField(
                n_harmonic_functions_xyz=n_harmonic_functions_xyz,
                n_harmonic_functions_dir=n_harmonic_functions_dir,
                n_hidden_neurons_xyz=n_hidden_neurons_xyz,
                n_hidden_neurons_dir=n_hidden_neurons_dir,
                n_layers_xyz=n_layers_xyz,
                append_xyz=append_xyz,
                view_dependency=view_dependency
            )

        self.camera_predictor_type = camera_predictor_type
        if camera_predictor_type == 'gt':
            self.camera_predictor = None
        elif camera_predictor_type == 'cnn':
            assert image_height == image_width
            self.camera_predictor = AzimuthElevationCameraPredictor(
                image_height,
                depth=depth_camera_predictor,
                channels=channels_camera_predictor,
                kernel_size=kernel_size_camera_predictor,
                replication_loss=replication_loss,
                replication_order=replication_order,
                northern_hemisphere=northern_hemisphere
            )
        elif camera_predictor_type == 'pe':
            self.camera_predictor = PEAzimuthElevationCameraPredictor(
                n_images,
                replication_loss=replication_loss,
                replication_order=replication_order,
                northern_hemisphere=northern_hemisphere
            )

        self._density_noise_std = density_noise_std
        self._chunk_size_test = chunk_size_test
        self._image_size = image_size
        self.visualization = visualization
        self.mask_loss = mask_loss
        self.replication_loss = replication_loss
        self.replication_order = replication_order

    def precache_rays(
        self,
        cache_cameras: List[CamerasBase],
        cache_camera_hashes: List[str],
    ):
        """
        Precaches the rays emitted from the list of cameras `cache_cameras`,
        where each camera is uniquely identified with the corresponding hash
        from `cache_camera_hashes`.

        The cached rays are moved to cpu and stored in
        `self._renderer['coarse']._ray_cache`.

        Raises `ValueError` when caching two cameras with the same hash.

        Args:
            cache_cameras: A list of `N` cameras for which the rays are pre-cached.
            cache_camera_hashes: A list of `N` unique identifiers for each
                camera from `cameras`.
        """
        self._renderer["coarse"].raysampler.precache_rays(
            cache_cameras,
            cache_camera_hashes,
        )

    def _process_ray_chunk(
        self,
        camera_hash: Optional[str],
        camera: CamerasBase,
        image: torch.Tensor,
        chunk_idx: int
    ) -> dict:
        """
        Samples and renders a chunk of rays.

        Args:
            camera_hash: A unique identifier of a pre-cached camera.
                If `None`, the cache is not searched and the sampled rays are
                calculated from scratch.
            camera: A batch of cameras from which the scene is rendered.
            image: A batch of corresponding ground truth images of shape
                ('batch_size', ·, ·, 3).
            chunk_idx: The index of the currently rendered ray chunk.
        Returns:
            out: `dict` containing the outputs of the rendering:
                `rgb_coarse`: The result of the coarse rendering pass.
                `rgb_fine`: The result of the fine rendering pass.
                `rgb_gt`: The corresponding ground-truth RGB values.
        """
        # Initialize the outputs of the coarse rendering to None.
        coarse_ray_bundle = None
        coarse_weights = None

        # First evaluate the coarse rendering pass, then the fine one.
        for renderer_pass in ("coarse", "fine"):
            (rgb, weights), ray_bundle_out = self._renderer[renderer_pass](
                cameras=camera,
                volumetric_function=self._implicit_function[renderer_pass],
                chunksize=self._chunk_size_test,
                chunk_idx=chunk_idx,
                density_noise_std=(self._density_noise_std if self.training else 0.0),
                input_ray_bundle=coarse_ray_bundle,
                ray_weights=coarse_weights,
                camera_hash=camera_hash,
            )

            if renderer_pass == "coarse":
                rgb_coarse = rgb
                # Store the weights and the rays of the first rendering pass
                # for the ensuing importance ray-sampling of the fine render.
                coarse_ray_bundle = ray_bundle_out
                coarse_weights = weights
                if image is not None:
                    # Sample the ground truth images at the xy locations of the
                    # rendering ray pixels.
                    batch_size = camera.R.shape[0]
                    image_sampled = torch.cat([image[..., :3]] * (batch_size // image.shape[0]), dim=0)
                    rgb_gt = sample_images_at_mc_locs(
                        image_sampled,
                        ray_bundle_out.xys,
                    )
                    mask = (torch.norm(image_sampled, dim=-1) > 1e-6)[..., None].float()
                    if self.mask_loss:
                        opacity_gt = sample_images_at_mc_locs(
                            mask,
                            ray_bundle_out.xys,
                        )[..., 0]
                else:
                    rgb_gt = None
                if self.mask_loss:
                    opacity_coarse = weights.sum(dim=-1)

            elif renderer_pass == "fine":
                rgb_fine = rgb
                if self.mask_loss:
                    opacity_fine = weights.sum(dim=-1)

            else:
                raise ValueError(f"No such rendering pass {renderer_pass}")

        out = {
            "rgb_fine": rgb_fine,
            "rgb_coarse": rgb_coarse,
            "rgb_gt": rgb_gt
        }
        if self.mask_loss:
            out["opacity_fine"] = opacity_fine
            out["opacity_coarse"] = opacity_coarse
            out["opacity_gt"] = opacity_gt
        if self.visualization:
            # Store the coarse rays/weights only for visualization purposes.
            out["coarse_ray_bundle"] = type(coarse_ray_bundle)(
                *[v.detach().cpu() for k, v in coarse_ray_bundle._asdict().items()]
            )
            out["coarse_weights"] = coarse_weights.detach().cpu()

        return out

    def forward(
        self,
        camera_hash: Optional[str],
        camera_gt: CamerasBase,
        image: torch.Tensor,
        align_gt: bool = False,
        alignment: Optional[torch.Tensor] = None
    ) -> Tuple[dict, dict, CamerasBase]:
        """
        Performs the coarse and fine rendering passes of the radiance field
        from the viewpoint of the input `camera`.
        Afterwards, both renders are compared to the input ground truth `image`
        by evaluating the peak signal-to-noise ratio and the mean-squared error.

        The rendering result depends on the `self.training` flag:
            - In the training mode (`self.training==True`), the function renders
              a random subset of image rays (Monte Carlo rendering).
            - In evaluation mode (`self.training==False`), the function renders
              the full image. In order to prevent out-of-memory errors,
              when `self.training==False`, the rays are sampled and rendered
              in batches of size `chunksize`.

        Args:
            camera_hash: A unique identifier of a pre-cached camera.
                If `None`, the cache is not searched and the sampled rays are
                calculated from scratch.
            camera_gt: A batch of cameras from which the scene is rendered.
            image: A batch of corresponding ground truth images of shape
                ('batch_size', ·, ·, 3).
            align_gt: Whether to use the aligne gt cameras.
            alignment: Alignment matrix.
        Returns:
            out: `dict` containing the outputs of the rendering:
                `rgb_coarse`: The result of the coarse rendering pass.
                `rgb_fine`: The result of the fine rendering pass.
                `rgb_gt`: The corresponding ground-truth RGB values.

                The shape of `rgb_coarse`, `rgb_fine`, `rgb_gt` depends on the
                `self.training` flag:
                    If `==True`, all 3 tensors are of shape
                    `(batch_size, n_rays_per_image, 3)` and contain the result
                    of the Monte Carlo training rendering pass.
                    If `==False`, all 3 tensors are of shape
                    `(batch_size, image_size[0], image_size[1], 3)` and contain
                    the result of the full image rendering pass.
            metrics: `dict` containing the error metrics comparing the fine and
                coarse renders to the ground truth:
                `mse_coarse`: Mean-squared error between the coarse render and
                    the input `image`
                `mse_fine`: Mean-squared error between the fine render and
                    the input `image`
                `psnr_coarse`: Peak signal-to-noise ratio between the coarse render and
                    the input `image`
                `psnr_fine`: Peak signal-to-noise ratio between the fine render and
                    the input `image`
        """
        if image.dim() == 3:
            image = image[None]
        else:
            image = image

        if self.camera_predictor is None:
            camera = camera_gt
        else:
            if not align_gt:
                if self.camera_predictor_type == 'cnn':
                    camera = self.camera_predictor(image, camera_gt)
                elif self.camera_predictor_type == 'pe':
                    camera = self.camera_predictor(camera_hash, camera_gt)
            else:
                camera = align_camera_gt(camera_gt, alignment)

        if not self.training:
            # Full evaluation pass.
            n_chunks = self._renderer["coarse"].raysampler.get_n_chunks(
                self._chunk_size_test,
                camera.R.shape[0],
            )
        else:
            # MonteCarlo ray sampling.
            n_chunks = 1

        # Process the chunks of rays.
        chunk_outputs = [
            self._process_ray_chunk(
                camera_hash,
                camera,
                image,
                chunk_idx,
            )
            for chunk_idx in range(n_chunks)
        ]

        if not self.training:
            # For a full render pass concatenate the output chunks,
            # and reshape to image size.
            out = {
                k: torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(-1, *self._image_size, channels)
                if chunk_outputs[0][k] is not None
                else None
                for k, channels in zip(
                    ("rgb_fine", "rgb_coarse", "rgb_gt", "opacity_fine", "opacity_coarse", "opacity_gt"),
                    (3, 3, 3, 1, 1, 1)
                )
            }
        else:
            out = chunk_outputs[0]

        # Calc the error metrics.
        metrics = {}
        if image is not None:
            for render_pass in ("coarse", "fine"):
                if self.replication_loss and not align_gt:
                    if render_pass == "coarse":
                        metrics[f"mse_{render_pass}"], activated_paths = calc_replication_loss(
                            out["rgb_" + render_pass][..., :3],
                            out["rgb_gt"][..., :3],
                            self.replication_order
                        )
                        selected_rgb_gt = select_paths(
                            out["rgb_gt"][..., :3], activated_paths, self.replication_order
                        )
                        out["rgb_selected_gt"] = selected_rgb_gt
                    selected_rgb = select_paths(
                        out["rgb_" + render_pass][..., :3], activated_paths, self.replication_order
                    )
                    out["rgb_selected_" + render_pass] = selected_rgb
                    if render_pass == "fine":
                        metrics[f"mse_{render_pass}"] = calc_mse(
                            out["rgb_selected_" + render_pass][..., :3],
                            out["rgb_selected_gt"][..., :3],
                        )
                    metrics[f"psnr_{render_pass}"] = calc_psnr(
                        selected_rgb,
                        selected_rgb_gt
                    )
                    if self.mask_loss:
                        # the model uses the paths founds for the photometric loss on the coarse model
                        selected_opacity = select_paths(
                            out["opacity_" + render_pass], activated_paths, self.replication_order
                        )
                        selected_opacity_gt = select_paths(
                            out["opacity_gt"], activated_paths, self.replication_order
                        )
                        metrics[f"mse_mask_{render_pass}"] = calc_mse(
                            selected_opacity,
                            selected_opacity_gt
                        )
                else:
                    for metric_name, metric_fun in zip(
                        ("mse", "psnr"), (calc_mse, calc_psnr)
                    ):
                        metrics[f"{metric_name}_{render_pass}"] = metric_fun(
                            out["rgb_" + render_pass][..., :3],
                            out["rgb_gt"][..., :3],
                        )
                    if self.mask_loss:
                        metrics[f"mse_mask_{render_pass}"] = calc_mse(
                            out["opacity_" + render_pass],
                            out["opacity_gt"]
                        )
            if self.replication_loss and not align_gt:
                batch_size = camera_gt.R.shape[0]
                camera = camera[activated_paths.cpu() * batch_size + torch.arange(batch_size)]

        return out, metrics, camera


def visualize_nerf_outputs(
        nerf_out: dict, output_cache: List, viz: Visdom, visdom_env: str, writer: SummaryWriter, steps: int
):
    """
    Visualizes the outputs of the `RadianceFieldRenderer`.

    Args:
        nerf_out: An output of the validation rendering pass.
        output_cache: A list with outputs of several training render passes.
        viz: A visdom connection object.
        visdom_env: The name of visdom environment for visualization.
        writer: A tensorboard writer object.
        steps: Number of steps.
    """

    # Show the training images.
    ims = torch.stack([o["image"] for o in output_cache])
    ims = torch.cat(list(ims), dim=1)
    viz.image(
        ims.permute(2, 0, 1),
        env=visdom_env,
        win="images",
        opts={"title": "train_images"},
    )
    writer.add_image("Train Images", ims.permute(2, 0, 1), steps)

    # Show the coarse and fine renders together with the ground truth images.
    ims_full = torch.cat(
        [
            nerf_out[imvar][0].permute(2, 0, 1).detach().cpu().clamp(0.0, 1.0)
            for imvar in ("rgb_coarse", "rgb_fine", "rgb_gt")
        ],
        dim=2,
    )
    viz.image(
        ims_full,
        env=visdom_env,
        win="images_full",
        opts={"title": "coarse | fine | target"},
    )
    writer.add_image("Coarse | Fine | Target", ims_full, steps)

    # Make a 3D plot of training cameras and their emitted rays.
    camera_trace = {
        f"camera_{ci:03d}": o["camera_pred"].cpu() for ci, o in enumerate(output_cache)
    }
    ray_pts_trace = {
        f"ray_pts_{ci:03d}": Pointclouds(
            ray_bundle_to_ray_points(o["coarse_ray_bundle"])
            .detach()
            .cpu()
            .view(1, -1, 3)
        )
        for ci, o in enumerate(output_cache)
    }
    plotly_plot = plot_scene(
        {
            "training_scene": {
                **camera_trace,
                **ray_pts_trace,
            },
        },
        pointcloud_max_points=5000,
        pointcloud_marker_size=1,
        camera_scale=0.3,
    )
    viz.plotlyplot(plotly_plot, env=visdom_env, win="scenes")

    # Tensorboard.

    # View directions.
    rotmat_gt = torch.cat([o["camera"].R for o in output_cache], 0)
    rotmat_pred = torch.cat([o["camera_pred"].R for o in output_cache], 0)
    alignment, rotmat_pred_aligned = align_rotation_set(rotmat_gt, rotmat_pred)
    xy_gt = rotmat_to_2d_coords(rotmat_gt)
    xy_pred_aligned = rotmat_to_2d_coords(rotmat_pred_aligned)
    xy_gt = xy_gt.cpu().detach().numpy()
    xy_pred_aligned = xy_pred_aligned.cpu().detach().numpy()

    fig = plt.figure(figsize=(6, 6), dpi=100)
    plt.plot(xy_gt[..., 0], xy_gt[..., 1], 'ro', label='gt')
    plt.plot(xy_pred_aligned[..., 0], xy_pred_aligned[..., 1], 'bo', label='pred (aligned)')
    for i in range(len(xy_gt)):
        plt.plot([xy_gt[i, 0], xy_pred_aligned[i, 0]], [xy_gt[i, 1], xy_pred_aligned[i, 1]], 'k--')
    theta = np.linspace(0.0, 2.0 * np.pi, 100)
    plt.plot((np.pi / 2.0) * np.cos(theta), (np.pi / 2.0) * np.sin(theta), '--', color='grey')
    plt.grid(True)
    plt.legend(loc="best")
    plt.axis('equal')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    writer.add_figure("View Directions (GT Frame)", fig, global_step=steps)

    return alignment
