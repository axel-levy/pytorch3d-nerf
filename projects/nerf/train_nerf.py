#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections
import os
import pickle
import warnings

import hydra
import numpy as np
import torch
from nerf.dataset import get_nerf_datasets, trivial_collate
from nerf.nerf_renderer import RadianceFieldRenderer, visualize_nerf_outputs
from nerf.stats import Stats
from omegaconf import DictConfig
from visdom import Visdom
from torch.utils.tensorboard import SummaryWriter


CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="lego")
def main(cfg):
    # Set the relevant seeds for reproducibility.
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        warnings.warn(
            "Please note that although executing on CPU is supported,"
            + "the training is unlikely to finish in reasonable time."
        )
        device = "cpu"

    # Load the training/validation data.
    train_dataset, val_dataset, _ = get_nerf_datasets(
        dataset_name=cfg.data.dataset_name,
        image_size=cfg.data.image_size,
    )

    # Initialize the Radiance Field model.
    model = RadianceFieldRenderer(
        image_size=cfg.data.image_size,
        n_pts_per_ray=cfg.raysampler.n_pts_per_ray,
        n_pts_per_ray_fine=cfg.raysampler.n_pts_per_ray,
        n_rays_per_image=cfg.raysampler.n_rays_per_image,
        min_depth=cfg.raysampler.min_depth,
        max_depth=cfg.raysampler.max_depth,
        stratified=cfg.raysampler.stratified,
        stratified_test=cfg.raysampler.stratified_test,
        chunk_size_test=cfg.raysampler.chunk_size_test,
        n_harmonic_functions_xyz=cfg.implicit_function.n_harmonic_functions_xyz,
        n_harmonic_functions_dir=cfg.implicit_function.n_harmonic_functions_dir,
        n_hidden_neurons_xyz=cfg.implicit_function.n_hidden_neurons_xyz,
        n_hidden_neurons_dir=cfg.implicit_function.n_hidden_neurons_dir,
        n_layers_xyz=cfg.implicit_function.n_layers_xyz,
        camera_predictor_type=cfg.camera_predictor.type,
        depth_camera_predictor=cfg.camera_predictor.depth,
        channels_camera_predictor=cfg.camera_predictor.channels,
        kernel_size_camera_predictor=cfg.camera_predictor.kernel_size,
        northern_hemisphere=cfg.camera_predictor.northern_hemisphere,
        n_images=len(train_dataset),
        density_noise_std=cfg.implicit_function.density_noise_std,
        visualization=cfg.visualization.visdom,
        view_dependency=cfg.implicit_function.view_dependency,
        mask_loss=cfg.loss_function.mask_loss,
        replication_loss=cfg.loss_function.replication_loss,
        replication_order=cfg.loss_function.replication_order
    )

    # Move the model to the relevant device.
    model.to(device)

    # Init stats to None before loading.
    stats = None
    optimizer_state_dict = None
    start_epoch = 0

    checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint_path)
    if len(cfg.checkpoint_path) > 0:
        # Make the root of the experiment directory.
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Resume training if requested.
        if cfg.resume and os.path.isfile(checkpoint_path):
            print('Resuming from checkpoint {}.'.format(checkpoint_path))
            loaded_data = torch.load(checkpoint_path)
            model.load_state_dict(loaded_data["model"])
            stats = pickle.loads(loaded_data["stats"])
            print('   => resuming from epoch {}.'.format(stats.epoch))
            optimizer_state_dict = loaded_data["optimizer"]
            start_epoch = stats.epoch

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.lr,
    )

    # Load the optimizer state dict in case we are resuming.
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        optimizer.last_epoch = start_epoch

    # Init the stats object.
    if stats is None:
        stat_list = ["loss", "mse_coarse", "mse_fine", "psnr_coarse", "psnr_fine", "sec/it"]
        if cfg.loss_function.mask_loss:
            stat_list += ["mse_mask_coarse", "mse_mask_fine"]
        stats = Stats(
            stat_list
        )

    # Learning rate scheduler setup.

    # Following the original code, we use exponential decay of the
    # learning rate: current_lr = base_lr * gamma ** (epoch / step_size)
    def lr_lambda(epoch):
        return cfg.optimizer.lr_scheduler_gamma ** (
            epoch / cfg.optimizer.lr_scheduler_step_size
        )

    # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_epoch - 1, verbose=False
    )

    # Initialize the cache for storing variables needed for visualization.
    visuals_cache = collections.deque(maxlen=cfg.visualization.history_size)

    # Init the visualization visdom env.
    if cfg.visualization.visdom:
        viz = Visdom(
            server=cfg.visualization.visdom_server,
            port=cfg.visualization.visdom_port,
            use_incoming_socket=False,
        )
    else:
        viz = None

    # Init tensorboard.
    summary_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.tensorboard.summary_dir)
    os.makedirs(summary_dir, exist_ok=True)
    writer = SummaryWriter(cfg.tensorboard.summary_dir)

    if cfg.data.precache_rays:
        # Precache the projection rays.
        model.eval()
        with torch.no_grad():
            for dataset in (train_dataset, val_dataset):
                cache_cameras = [e["camera"].to(device) for e in dataset]
                cache_camera_hashes = [e["camera_idx"] for e in dataset]
                model.precache_rays(cache_cameras, cache_camera_hashes)

    print("Building training dataloader...")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=trivial_collate,
    )

    # The validation dataloader is just an endless stream of random samples.
    print("Building testing dataloader...")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=trivial_collate,
        sampler=torch.utils.data.RandomSampler(
            val_dataset,
            replacement=True,
            num_samples=cfg.optimizer.max_epochs,
        ),
    )

    # Set the model to the training mode.
    model.train()

    print("Training starts now.")
    # Run the main training loop.
    alignment = torch.eye(3, dtype=torch.float32, device=device)
    for epoch in range(start_epoch, cfg.optimizer.max_epochs):
        stats.new_epoch()  # Init a new epoch.
        for iteration, batch in enumerate(train_dataloader):
            image, mask, camera, camera_idx = batch[0].values()
            image = image.to(device)
            camera = camera.to(device)

            optimizer.zero_grad()

            # Run the forward pass of the model.
            nerf_out, metrics, camera_pred = model(
                camera_idx if cfg.data.precache_rays or cfg.camera_predictor.type == 'pe' else None,
                camera,
                image,
            )

            # The loss is a sum of coarse and fine MSEs
            loss = metrics["mse_coarse"] + metrics["mse_fine"]

            if cfg.loss_function.mask_loss:
                loss += cfg.loss_function.mask_loss_weight * (metrics["mse_mask_coarse"] + metrics["mse_mask_fine"])

            # Take the training step.
            loss.backward()
            optimizer.step()

            # Update stats with the current metrics.
            stats.update(
                {"loss": float(loss), **metrics},
                stat_set="train"
            )

            if iteration % cfg.stats_print_interval == 0:
                stats.print(stat_set="train")

            # Update the visualization cache.
            if viz is not None:
                visuals_cache.append(
                    {
                        "camera": camera.cpu(),
                        "camera_pred": camera_pred.cpu(),
                        "camera_idx": camera_idx,
                        "image": image.cpu().detach(),
                        "rgb_fine": nerf_out["rgb_fine"].cpu().detach(),
                        "rgb_coarse": nerf_out["rgb_coarse"].cpu().detach(),
                        "rgb_gt": nerf_out["rgb_gt"].cpu().detach(),
                        "coarse_ray_bundle": nerf_out["coarse_ray_bundle"],
                    }
                )

        # Adjust the learning rate.
        lr_scheduler.step()

        # Validation
        if epoch % cfg.validation_epoch_interval == 0 and epoch > 0:

            # Sample a validation camera/image.
            val_batch = next(val_dataloader.__iter__())
            val_image, val_mask, val_camera, camera_idx = val_batch[0].values()
            val_image = val_image.to(device)
            val_camera = val_camera.to(device)

            # Activate eval mode of the model (lets us do a full rendering pass).
            model.eval()
            with torch.no_grad():
                val_nerf_out, val_metrics, val_camera_pred = model(
                    camera_idx if cfg.data.precache_rays else None,
                    val_camera,
                    val_image,
                    align_gt=True,
                    alignment=alignment
                )

            # Update stats with the validation metrics.
            stats.update(val_metrics, stat_set="val")
            stats.print(stat_set="val")

            if viz is not None:
                # Plot that loss curves into visdom.
                stats.plot_stats(
                    viz=viz,
                    visdom_env=cfg.visualization.visdom_env,
                    plot_file=None,
                    writer=writer
                )
                # Visualize the intermediate results.
                alignment = visualize_nerf_outputs(
                    val_nerf_out, visuals_cache, viz, cfg.visualization.visdom_env, writer, epoch
                )

            # Set the model back to train mode.
            model.train()

        # Checkpoint.
        if (
            epoch % cfg.checkpoint_epoch_interval == 0
            and len(cfg.checkpoint_path) > 0
            and epoch > 0
        ):
            print('Storing checkpoint {}.'.format(checkpoint_path))
            data_to_store = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "stats": pickle.dumps(stats),
            }
            torch.save(data_to_store, checkpoint_path)


if __name__ == "__main__":
    main()
