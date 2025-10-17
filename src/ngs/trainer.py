"""Training module from gsplat."""

import json
import math
import os
import time
from collections import defaultdict

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from torchmetrics.image import StructuralSimilarityIndexMeasure
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import assert_never

from ngs.config import TrainerConfig as Config
from ngs.ngs_strategy import DefaultStrategy
from ngs.data.colmap import Dataset, Parser
from ngs.data.traj import generate_ellipse_path_z, generate_interpolated_path, generate_spherical_poses, generate_spiral_path
from ngs.utils.lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss
from ngs.utils.noise import (
    add_noise_splats_and_process_convex_hull,
    add_noise_splats_coarse_to_fine,
    compute_convex_hull_mesh,
    remove_noise_splats,
    reset_color_noise_splats,
    reset_opacity_noise_splats,
    identify_noise_gaussians_to_prune,
)
from ngs.utils.utils import (
    AppearanceOptModule,
    CameraOptModule,
    knn,
    rgb_to_sh,
    save_ply,
    set_random_seed,
)



def erode_masks(masks, kernel_size=3, iterations=1):
    """Apply erosion to masks using max pooling with negative values."""
    import torch.nn.functional as F

    # Create padding to maintain size
    padding = kernel_size // 2
    eroded = masks

    for _ in range(iterations):
        # Invert mask (1->0, 0->1), apply max pooling, then invert back
        inverted = 1 - eroded
        pooled = F.max_pool2d(inverted.float(), kernel_size=kernel_size, stride=1, padding=padding)
        eroded = 1 - pooled

    return eroded


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: int | None = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> tuple[torch.nn.ParameterDict, dict[str, torch.optim.Optimizer], int]:
    """Initialize Gaussian splats and their optimizers.

    Args:
        parser: Parser object containing scene information
        init_type: Type of initialization ('sfm' or 'random')
        init_num_pts: Number of points for random initialization
        init_extent: Extent of random initialization volume
        init_opacity: Initial opacity value
        init_scale: Scale factor for gaussian size
        scene_scale: Global scene scale factor
        sh_degree: Spherical harmonics degree
        sparse_grad: Whether to use sparse gradients
        visible_adam: Whether to use visible Adam optimizer
        batch_size: Batch size for scaling learning rates
        feature_dim: Feature dimension for appearance embedding
        device: Device to put tensors on
        world_rank: Rank in distributed training
        world_size: World size for distributed training

    Returns:
        Tuple of (ParameterDict of gaussian parameters, dict of optimizers)
    """
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(self, local_rank: int, world_rank, world_size: int, cfg: Config) -> None:
        """Initialize the Runner class."""
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            optimize_foreground=cfg.optimize_foreground,
            load_images_in_memory=cfg.load_images_in_memory,
        )


        self.num_noise_gaussians = 0
        self.noise_gaussian_data = None
        self.convex_hull_data = None
        self.voxel_resolution = cfg.voxel_resolution
        self.filter_hull = None
        self.voxel_extent = None
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = cfg.feature_dim if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of trainable GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(scene_scale=self.scene_scale)
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            if feature_dim is None:
                raise ValueError("feature_dim is None")
            self.app_module = AppearanceOptModule(len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree).to(
                self.device
            )
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        undist_masks: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict]:
        """Rasterize the splats."""
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            # Process all features through app_module
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            if self.num_noise_gaussians > 0:
                colors[:, self.num_noise_gaussians :] = self.splats["colors"][self.num_noise_gaussians :]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(self.cfg.strategy.absgrad if isinstance(self.cfg.strategy, DefaultStrategy) else False),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            **kwargs,
        )
        return render_colors, render_alphas, info

    @torch.no_grad()
    def save_for_blender(self, step: int):
        """Save model with all info plus Blender-compatible colors."""
        save_dir = f"{self.cfg.result_dir}/blender_export"
        os.makedirs(save_dir, exist_ok=True)

        if self.num_noise_gaussians > 0:
            filtered_splats = {k: v[self.num_noise_gaussians :] for k, v in self.splats.items()}
            inside_splats = {k: v[: self.num_noise_gaussians] for k, v in self.splats.items()}
            all_splats = {k: v[:] for k, v in self.splats.items()}

        else:
            filtered_splats = self.splats
            inside_splats = None

        save_ply(
            gaussian_data=filtered_splats,
            filename=f"{save_dir}/gaussians_{step}.ply",
            device=self.device,
        )
        if inside_splats is not None:
            save_ply(
                gaussian_data=inside_splats,
                filename=f"{save_dir}/gaussians_inside_{step}.ply",
                device=self.device,
            )
            save_ply(
                gaussian_data=all_splats,
                filename=f"{save_dir}/gaussians_all_{step}.ply",
                device=self.device,
            )

    def train(self):
        """Train the model."""
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)),
        ]

        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(torch.optim.lr_scheduler.ExponentialLR(self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)))
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)),
                    ]
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            persistent_workers=False,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        freeze_surface = False
        freeze_noise = False
        # Training loop.
        global_tic = time.time()
        end_step = cfg.unfreeze_steps[0] + cfg.noise_steps[0] if cfg.noise_steps[0] > 0 else cfg.max_steps
        pbar = tqdm.tqdm(range(init_step, end_step))
        for step in pbar:
            if step == cfg.noise_steps[0]:
                freeze_surface = True
                # Initialize noise Gaussians at multiple resolutions and prune at each step
                self.initialize_and_prune_noise_gaussians(
                    erosion_steps=cfg.erosion_steps, increase_prune_steps=cfg.increase_erosion_steps
                )
                reset_opacity_noise_splats(
                    self.splats, self.optimizers, self.num_noise_gaussians, opacity=cfg.init_noise_opa, device=device
                )
                reset_color_noise_splats(
                    self.splats, self.optimizers, self.num_noise_gaussians, noise_color=cfg.noise_color, device=device
                )
                self.save_for_blender(step)
            self.train_loop(step, pbar, trainloader, trainloader_iter, global_tic, schedulers, freeze_surface, freeze_noise)

        if cfg.noise_steps[0] > 0:
            # Reset learning rate scheduler for means
            if cfg.reset_lr:
                self.optimizers["means"].param_groups[0]["lr"] = 1.6e-4
                if len(schedulers) > 0:
                    schedulers[0] = torch.optim.lr_scheduler.ExponentialLR(
                        self.optimizers["means"], gamma=0.01 ** (1.0 / (cfg.max_steps - step))
                    )
                else:
                    schedulers.append(
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.optimizers["means"], gamma=0.01 ** (1.0 / (cfg.max_steps - step))
                        )
                    )
            print(f"Reset learning rate scheduler at step {step}")
            if len(cfg.noise_steps) == 2 and cfg.noise_steps[1] > 0:
                end_step = cfg.noise_steps[1] + cfg.unfreeze_steps[0]
            else:
                end_step = cfg.max_steps

            pbar = tqdm.tqdm(range(cfg.noise_steps[0] + 1, end_step))
            freeze_surface = False
            freeze_noise = True
            reset_opacity_noise_splats(
                self.splats, self.optimizers, self.num_noise_gaussians, opacity=cfg.freeze_noise_opa, device=device
            )
            reset_color_noise_splats(
                self.splats, self.optimizers, self.num_noise_gaussians, noise_color=cfg.noise_color, device=device
            )
            for step in pbar:
                self.train_loop(step, pbar, trainloader, trainloader_iter, global_tic, schedulers, freeze_surface, freeze_noise)

            if len(cfg.noise_steps) == 2 and cfg.noise_steps[1] > 0:
                end_step = cfg.noise_steps[1] + cfg.unfreeze_steps[0]
                pbar = tqdm.tqdm(range(cfg.noise_steps[1], end_step))

                remove_noise_splats(self.splats, self.optimizers, self.num_noise_gaussians, device=device)
                self.num_noise_gaussians = 0
                self.noise_gaussian_data = None
                self.voxel_resolution = cfg.voxel_resolution
                freeze_noise = False
                freeze_surface = True
                self.initialize_and_prune_noise_gaussians(
                    erosion_steps=cfg.erosion_steps, increase_prune_steps=cfg.increase_erosion_steps
                )
                reset_opacity_noise_splats(
                    self.splats, self.optimizers, self.num_noise_gaussians, opacity=cfg.init_noise_opa, device=device
                )
                reset_color_noise_splats(
                    self.splats, self.optimizers, self.num_noise_gaussians, noise_color=cfg.noise_color, device=device
                )
                self.save_for_blender(step)

                for step in pbar:
                    self.train_loop(
                        step, pbar, trainloader, trainloader_iter, global_tic, schedulers, freeze_surface, freeze_noise
                    )

                pbar = tqdm.tqdm(range(end_step, cfg.max_steps))
                freeze_noise = True
                freeze_surface = False
                reset_opacity_noise_splats(
                    self.splats, self.optimizers, self.num_noise_gaussians, opacity=cfg.freeze_noise_opa, device=device
                )
                reset_color_noise_splats(
                    self.splats, self.optimizers, self.num_noise_gaussians, noise_color=cfg.noise_color, device=device
                )
                print(f"Unfroze surface Gaussians at step {step}")

                for step in pbar:
                    self.train_loop(
                        step, pbar, trainloader, trainloader_iter, global_tic, schedulers, freeze_surface, freeze_noise
                    )

    def train_loop(self, step, pbar, trainloader, trainloader_iter, global_tic, schedulers, freeze_surface, freeze_noise):
        """Training loop."""
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size
        if not cfg.disable_viewer:
            while self.viewer.state.status == "paused":
                time.sleep(0.01)
            self.viewer.lock.acquire()
            tic = time.time()

        try:
            data = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(trainloader)
            data = next(trainloader_iter)

        camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
        Ks = data["K"].to(device)  # [1, 3, 3]
        pixels = data["image"].to(device)  # [1, H, W, 3]
        num_train_rays_per_step = pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
        image_ids = data["image_id"].to(device)
        masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
        undist_masks = data["undist_mask"].to(device) if "undist_mask" in data else None  # [1, H, W]
        if cfg.depth_loss:
            points = data["points"].to(device)  # [1, M, 2]
            depths_gt = data["depths"].to(device)  # [1, M]

        height, width = pixels.shape[1:3]

        if cfg.pose_noise:
            camtoworlds = self.pose_perturb(camtoworlds, image_ids)

        if cfg.pose_opt:
            camtoworlds = self.pose_adjust(camtoworlds, image_ids)

        # sh schedule
        sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)


        # New noise initialization logic (all at once at the first noise step)

        with torch.no_grad():
            if self.filter_hull is None and step == cfg.hull_filter_step:
                self.filter_hull, _, _ = compute_convex_hull_mesh(self.splats["means"], scale_factor=1.1)

            # Periodically reset noise gaussian properties if they exist
            if self.num_noise_gaussians > 0 and cfg.noise_color is None:
                reset_color_noise_splats(self.splats, self.optimizers, self.num_noise_gaussians, device=device)

        # forward
        renders, alphas, info = self.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree_to_use,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            image_ids=image_ids,
            render_mode="RGB+ED" if cfg.depth_loss else "RGB",
            undist_masks=undist_masks,
            distributed=self.world_size > 1,
        )
        if renders.shape[-1] == 4:
            colors, depths = renders[..., 0:3], renders[..., 3:4]
        else:
            colors, depths = renders, None

        if cfg.use_bilateral_grid:
            grid_y, grid_x = torch.meshgrid(
                (torch.arange(height, device=self.device) + 0.5) / height,
                (torch.arange(width, device=self.device) + 0.5) / width,
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
            colors = slice(self.bil_grids, grid_xy, colors, image_ids)["rgb"]

        if cfg.random_bkgd:
            # Define 6 fixed RGB color tuples
            color_choices = torch.tensor(
                [
                    [1.0, 0.0, 0.0],  # Red
                    [0.0, 1.0, 0.0],  # Green
                    [0.0, 0.0, 1.0],  # Blue
                    [1.0, 1.0, 0.0],  # Yellow
                    [1.0, 0.0, 1.0],  # Magenta
                    [0.0, 1.0, 1.0],  # Cyan
                ],
                device=device,
            )

            # Randomly choose one of the 6 colors for each pixel
            H, W = colors.shape[1:3]
            random_indices = torch.randint(0, 6, (1, H, W), device=device)
            bkgd = color_choices[random_indices]  # [1, H, W, 3]
            if masks is not None:
                bkgd[masks < 0.5] = 0.0
            colors = alphas * colors + bkgd * (1.0 - alphas)
        # else:
        #     colors = alphas * colors + (1.0 - alphas) * torch.ones_like(colors)

        self.cfg.strategy.step_pre_backward(
            params=self.splats,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=step,
            info=info,
        )

        if masks is not None:
            # Get indices of True values, handling batch dimension
            batch_indices, y_indices, x_indices = torch.where(masks > 0.1)

            # Get min/max coordinates for each batch
            x_min = x_indices.min().item()
            x_max = x_indices.max().item()
            y_min = y_indices.min().item()
            y_max = y_indices.max().item()

            colors = colors[:, y_min:y_max, x_min:x_max]
            pixels = pixels[:, y_min:y_max, x_min:x_max]

        #  # Suppress pure white pixels
        # white_mask = (pixels >= 1.0 - 1e-6).all(dim=-1)  # Find pixels where all RGB channels are 1.0

        # # Option 1: Set white pixels to black (or any other color)
        # pixels[white_mask] = 0.0  # Set to black
        # colors[white_mask] = 0.0  # Set to black

        l1loss = F.l1_loss(colors, pixels)

        # tile_loss_info = None  # compute_tile_l1_loss(colors, pixels, tile_size=16)
        ssimloss = 1.0 - self.ssim(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
        loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
        if cfg.depth_loss:
            # query depths from depth map
            points = torch.stack(
                [
                    points[:, :, 0] / (width - 1) * 2 - 1,
                    points[:, :, 1] / (height - 1) * 2 - 1,
                ],
                dim=-1,
            )  # normalize to [-1, 1]
            grid = points.unsqueeze(2)  # [1, M, 1, 2]
            depths = F.grid_sample(depths.permute(0, 3, 1, 2), grid, align_corners=True)  # [1, 1, M, 1]
            depths = depths.squeeze(3).squeeze(1)  # [1, M]
            # calculate loss in disparity space
            disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
            disp_gt = 1.0 / depths_gt  # [1, M]
            depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
            loss += depthloss * cfg.depth_lambda
        if cfg.use_bilateral_grid:
            tvloss = 10 * total_variation_loss(self.bil_grids.grids)
            loss += tvloss

        # segmentation loss
        if masks is not None:
            segmentation_loss = torch.sum(alphas * (1.0 - masks.unsqueeze(-1))) / ((1.0 - masks).sum())
            loss += segmentation_loss

            if cfg.foreground_loss:
                masks = erode_masks(masks, kernel_size=3, iterations=1)
                foreground_loss = torch.sum((1.0 - alphas) * masks.unsqueeze(-1)) / masks.sum()
                loss += foreground_loss

    
        # regularizations
        if cfg.opacity_reg > 0.0:
            loss = loss + cfg.opacity_reg * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
        if cfg.scale_reg > 0.0:
            loss = loss + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()

        loss.backward()

        # Zero out gradients for noise splats
        with torch.no_grad():
            # Zero gradients for colors (handle both SH and feature cases)
            if "sh0" in self.splats and self.splats["sh0"].grad is not None:
                self.splats["sh0"].grad[: self.num_noise_gaussians] = 0
            if "shN" in self.splats and self.splats["shN"].grad is not None:
                self.splats["shN"].grad[: self.num_noise_gaussians] = 0
            if "colors" in self.splats and self.splats["colors"].grad is not None:
                self.splats["colors"].grad[: self.num_noise_gaussians] = 0
            if "features" in self.splats and self.splats["features"].grad is not None:
                self.splats["features"].grad[: self.num_noise_gaussians] = 0
            # Zero gradients for means
            if "means" in self.splats and self.splats["means"].grad is not None:
                self.splats["means"].grad[: self.num_noise_gaussians] = 0
            # Zero gradients for quats
            if "quats" in self.splats and self.splats["quats"].grad is not None:
                self.splats["quats"].grad[: self.num_noise_gaussians] = 0
            # Zero gradients for scales
            if self.splats["scales"].grad is not None:
                self.splats["scales"].grad[: self.num_noise_gaussians] = 0

            if freeze_noise:
                if "opacities" in self.splats and self.splats["opacities"].grad is not None:
                    self.splats["opacities"].grad[: self.num_noise_gaussians] = 0

            if freeze_surface:
                for k in self.splats.keys():
                    if self.splats[k].grad is not None:
                        self.splats[k].grad[self.num_noise_gaussians :] = 0

        desc = f"loss={loss.item():.3f}| sh degree={sh_degree_to_use}| "
        if cfg.depth_loss:
            desc += f"depth loss={depthloss.item():.6f}| "
        if cfg.pose_opt and cfg.pose_noise:
            # monitor the pose error if we inject noise
            pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
            desc += f"pose err={pose_err.item():.6f}| "
        pbar.set_description(desc)

        # write images (gt and render)
        # if world_rank == 0 and step % 800 == 0:
        #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
        #     canvas = canvas.reshape(-1, *canvas.shape[2:])
        #     imageio.imwrite(
        #         f"{self.render_dir}/train_rank{self.world_rank}.png",
        #         (canvas * 255).astype(np.uint8),
        #     )

        # save checkpoint before updating the model
        if step in [i - 1 for i in cfg.save_steps] or step == cfg.max_steps - 1:
            mem = torch.cuda.max_memory_allocated() / 1024**3
            stats = {
                "mem": mem,
                "ellipse_time": time.time() - global_tic,
                "num_GS": len(self.splats["means"]),
            }
            print("Step: ", step, stats)
            with open(
                f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                "w",
            ) as f:
                json.dump(stats, f)
            data = {"step": step, "splats": self.splats.state_dict()}
            data["num_noise_gaussians"] = self.num_noise_gaussians
            data["noise_gaussian_data"] = self.noise_gaussian_data
            if cfg.pose_opt:
                if world_size > 1:
                    data["pose_adjust"] = self.pose_adjust.module.state_dict()
                else:
                    data["pose_adjust"] = self.pose_adjust.state_dict()
            if cfg.app_opt:
                if world_size > 1:
                    data["app_module"] = self.app_module.module.state_dict()
                else:
                    data["app_module"] = self.app_module.state_dict()
            torch.save(data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt")

        # Turn Gradients into Sparse Tensor before running optimizer
        if cfg.sparse_grad:
            if not cfg.packed:
                raise ValueError("Sparse gradients only work with packed mode.")
            gaussian_ids = info["gaussian_ids"]
            for k in self.splats.keys():
                grad = self.splats[k].grad
                if grad is None or grad.is_sparse:
                    continue
                self.splats[k].grad = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=grad[gaussian_ids],  # [nnz, ...]
                    size=self.splats[k].size(),  # [N, ...]
                    is_coalesced=len(Ks) == 1,
                )

        if cfg.visible_adam:
            self.splats.means.shape[0]
            if cfg.packed:
                visibility_mask = torch.zeros_like(self.splats["opacities"], dtype=bool)
                visibility_mask.scatter_(0, info["gaussian_ids"], 1)
            else:
                visibility_mask = (info["radii"] > 0).any(0)

        # optimize
        for optimizer in self.optimizers.values():
            if cfg.visible_adam:
                optimizer.step(visibility_mask)
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for optimizer in self.pose_optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if freeze_surface:
            for optimizer in self.app_optimizers:
                optimizer.zero_grad(set_to_none=True)
        else:
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        for optimizer in self.bil_grid_optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers:
            scheduler.step()

        # Run post-backward steps after backward and optimizer
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.num_noise_gaussians, self.noise_gaussian_data = self.cfg.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=cfg.packed,
                num_noise=self.num_noise_gaussians,
                freeze_surface=freeze_surface,
                noise_gaussian_data=self.noise_gaussian_data,
                filter_hull=self.filter_hull,
            )
        else:
            assert_never(self.cfg.strategy)

        if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
            if freeze_surface:
                self.writer.add_scalar("train/num_noise_GS", self.num_noise_gaussians, step - cfg.noise_steps[0])
            else:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]) - self.num_noise_gaussians, step)
                self.writer.add_scalar("train/mem", mem, step)
                self.writer.add_histogram(
                    "train/scales", torch.exp(self.splats["scales"][self.num_noise_gaussians :]).max(dim=-1).values, step
                )
                self.writer.add_histogram(
                    "train/opacities", torch.sigmoid(self.splats["opacities"][self.num_noise_gaussians :]), step
                )
                if masks is not None:
                    self.writer.add_scalar("train/segmentation_loss", segmentation_loss.item(), step)
                    if cfg.foreground_loss:
                        self.writer.add_scalar("train/foreground_loss", foreground_loss.item(), step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)

            self.writer.flush()

        # eval the full set
        if step in [i - 1 for i in cfg.eval_steps]:
            self.eval(step)
            self.save_for_blender(step)


        # run compression
        if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
            self.run_compression(step=step)

        if not cfg.disable_viewer:
            self.viewer.lock.release()
            num_train_steps_per_sec = 1.0 / (time.time() - tic)
            num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
            # Update the viewer state.
            self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
            # Update the scene.
            self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def initialize_and_prune_noise_gaussians(self, erosion_steps, increase_prune_steps=True):
        """Initialize noise Gaussians at multiple resolutions with pruning and deletion at each stage."""
        device = self.device
        cfg = self.cfg

        print("Starting multi-resolution noise Gaussian initialization and pruning...")

        # Start with initial resolution
        current_resolution = self.voxel_resolution

        # Initial noise Gaussian addition
        print(f"Step 1: Adding initial noise Gaussians at resolution {current_resolution}...")
        (
            self.splats,
            self.optimizers,
            self.num_noise_gaussians,
            self.noise_gaussian_data,
            self.convex_hull_data,
        ) = add_noise_splats_and_process_convex_hull(
            self.splats,
            self.optimizers,
            current_resolution,
            device,
            0.01,
            0.5,
            cfg.opa_threshold,
            cfg.opaque_surface,
        )
        print(f"Added {self.num_noise_gaussians} initial noise Gaussians")
        self.voxel_extent = torch.abs(self.convex_hull_data[2].to(device).float() - self.convex_hull_data[1].to(device).float())
        self.voxel_extent = torch.min(self.voxel_extent)
        print(f"Voxel extent: {self.voxel_extent}")
        # Prune and manually remove in each stage
        self._prune_and_remove_noise_gaussians(current_resolution, erosion_steps)

        # Perform coarse-to-fine refinement
        for i in range(3):
            # Double the resolution each time
            current_resolution *= 2
            self.voxel_resolution = current_resolution
            if increase_prune_steps:
                erosion_steps += 1

            print(f"Step {i+2}: Adding finer noise Gaussians at resolution {current_resolution}...")
            (
                self.splats,
                self.optimizers,
                num_added,
                self.noise_gaussian_data,
                self.convex_hull_data,
            ) = add_noise_splats_coarse_to_fine(
                self.splats,
                self.optimizers,
                current_resolution,
                device,
                0.001,
                0.5,
                cfg.opa_threshold,
                self.noise_gaussian_data,
                self.convex_hull_data,
            )
            self.num_noise_gaussians += num_added
            print(f"Added {num_added} fine noise Gaussians")

            # Prune and manually remove at current resolution
            self._prune_and_remove_noise_gaussians(current_resolution, erosion_steps)

        print(f"Multi-resolution noise initialization complete. Total noise Gaussians: {self.num_noise_gaussians}")

    @torch.no_grad()
    def _prune_and_remove_noise_gaussians(self, current_resolution, erosion_steps):
        """Prune noise Gaussians across dataset and manually remove them."""
        print("Pruning noise Gaussians across all training views...")
        device = self.device
        cfg = self.cfg

        # Create a dataloader that iterates through all training images
        full_trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        # Track which noise Gaussians should be pruned
        cumulative_prune_mask = torch.zeros(len(self.splats["means"]), dtype=torch.bool, device=device)

        # First do depth-based pruning
        for data_batch in tqdm.tqdm(full_trainloader, desc="Depth-based pruning"):
            img_camtoworlds = data_batch["camtoworld"].to(device)
            img_Ks = data_batch["K"].to(device)
            img_height, img_width = data_batch["image"].shape[1:3]

            view_prune_mask = identify_noise_gaussians_to_prune(
                means=self.splats["means"],
                quats=self.splats["quats"],
                scales=torch.exp(self.splats["scales"]),
                opacities=torch.sigmoid(self.splats["opacities"]),
                viewmats=torch.linalg.inv(img_camtoworlds),
                Ks=img_Ks,
                width=img_width,
                height=img_height,
                num_noise_gaussians=self.num_noise_gaussians,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                downsample_factor=4.0,
            )

            cumulative_prune_mask = cumulative_prune_mask | view_prune_mask
            del view_prune_mask
            torch.cuda.empty_cache()

        num_pruned_depth = cumulative_prune_mask.sum().item()
        print(
            f"Depth pruning removed {num_pruned_depth} noise Gaussians ({100 * num_pruned_depth / self.num_noise_gaussians:.2f}%)"
        )

        # Now do erosion-based pruning if we have noise gaussians left
        if self.num_noise_gaussians > 0 and self.noise_gaussian_data is not None:
            print("Performing erosion-based pruning...")

            # Create temporary noise_gaussian_data for remaining gaussians after depth pruning
            depth_keep_mask = ~cumulative_prune_mask[: self.num_noise_gaussians]
            temp_noise_data = self.noise_gaussian_data[depth_keep_mask.cpu()]

            if len(temp_noise_data) > 0:  # Only proceed if we have gaussians left after depth pruning
                # Create occupancy grid at current resolution
                occupancy = torch.zeros((current_resolution,) * 3, dtype=torch.bool, device=device)

                # Efficient vectorized grid filling from add_noise_splats_coarse_to_fine
                orig_resolutions = temp_noise_data[:, 0]
                grid_coords = temp_noise_data[:, 1:].long()
                scale_factors = current_resolution / orig_resolutions
                base_coords = (grid_coords.float() * scale_factors[:, None]).long()

                for unique_scale in torch.unique(scale_factors):
                    scale_int = int(unique_scale.item())
                    mask = scale_factors == unique_scale
                    curr_base_coords = base_coords[mask]

                    if len(curr_base_coords) == 0:
                        continue

                    offsets = torch.tensor(
                        [[x, y, z] for x in range(scale_int) for y in range(scale_int) for z in range(scale_int)], device=device
                    )
                    curr_fine_coords = curr_base_coords.unsqueeze(1) + offsets.unsqueeze(0)
                    curr_fine_coords = curr_fine_coords.reshape(-1, 3)

                    valid_mask = ((curr_fine_coords >= 0) & (curr_fine_coords < current_resolution)).all(dim=1)
                    valid_coords = curr_fine_coords[valid_mask]

                    if len(valid_coords) > 0:
                        unique_coords = torch.unique(valid_coords, dim=0)
                        occupancy[unique_coords[:, 0], unique_coords[:, 1], unique_coords[:, 2]] = True

                # Perform erosion with multiple iterations based on scale_multiplier
                kernel_size = 3
                padsize = kernel_size // 2
                kernel = torch.ones((kernel_size,) * 3, dtype=torch.bool, device=device)
                weight = kernel.float().view(1, 1, kernel_size, kernel_size, kernel_size)

                # Number of iterations scales with scale_multiplier
                # If scale_multiplier is 2, we do 2 iterations, if it's 4, we do 4 iterations, etc.
                num_iterations = int(erosion_steps)
                print(f"Performing {num_iterations} erosion iterations...")

                eroded = occupancy.clone()
                for i in range(num_iterations):
                    padded = torch.nn.functional.pad(eroded.float(), (padsize,) * 6, mode="constant", value=0)

                    neighbor_count = torch.nn.functional.conv3d(padded.view(1, 1, *padded.shape), weight, padding=0).squeeze()

                    eroded = neighbor_count >= kernel.sum()

                    # Optional: print progress for long iterations
                    if num_iterations > 3:
                        print(f"Completed erosion iteration {i+1}/{num_iterations}")

                # Only check erosion for current resolution gaussians
                curr_res_mask = temp_noise_data[:, 0] == current_resolution
                if curr_res_mask.any():
                    curr_res_coords = temp_noise_data[curr_res_mask, 1:].long()

                    # Check which current resolution gaussians should be pruned
                    pruned = ~eroded[curr_res_coords[:, 0], curr_res_coords[:, 1], curr_res_coords[:, 2]]

                    # Create erosion pruning mask for all noise gaussians
                    erosion_prune_mask = torch.zeros(self.num_noise_gaussians, dtype=torch.bool, device=device)

                    # Map back to original indices
                    curr_res_indices = torch.where(depth_keep_mask)[0][curr_res_mask]
                    erosion_prune_mask[curr_res_indices] = pruned

                    # Add erosion pruning to cumulative mask
                    cumulative_prune_mask[: self.num_noise_gaussians] |= erosion_prune_mask

                    num_pruned_erosion = erosion_prune_mask.sum().item()
                    print(f"Erosion pruning removed {num_pruned_erosion} noise Gaussians at resolution {current_resolution}")

        # Now do the final removal of all pruned gaussians
        if cumulative_prune_mask.any():
            print("Removing all pruned Gaussians...")
        keep_indices = torch.where(~cumulative_prune_mask)[0]

        # Update all tensors in splats
        for k, param in self.splats.items():
            new_param = param.data[keep_indices].clone()
            self.splats[k] = torch.nn.Parameter(new_param)

            if k in self.optimizers:
                old_optimizer = self.optimizers[k]
                old_param_group = old_optimizer.param_groups[0]
                lr = old_param_group["lr"]
                name = old_param_group.get("name", k)

                self.optimizers[k] = type(old_optimizer)(
                    [{"params": self.splats[k], "lr": lr, "name": name}],
                    eps=old_optimizer.defaults.get("eps", 1e-8),
                    betas=old_optimizer.defaults.get("betas", (0.9, 0.999)),
                )

            # Update noise gaussian data and count
        if self.noise_gaussian_data is not None:
            noise_indices = keep_indices[keep_indices < self.num_noise_gaussians]
            self.noise_gaussian_data = self.noise_gaussian_data[noise_indices.cpu()]
            self.num_noise_gaussians = len(noise_indices)

        print(f"After removal: {self.num_noise_gaussians} noise Gaussians remain")

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank

        valloader = torch.utils.data.DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=0)
        if len(valloader) == 0:
            return
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device)
            undist_masks = data["undist_mask"].to(device) if "undist_mask" in data else None
            height, width = pixels.shape[1:3]
            masks = data["mask"].to(device) if "mask" in data else None

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                undist_masks=undist_masks,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]
            if masks is not None:
                # Get indices of True values, handling batch dimension
                batch_indices, y_indices, x_indices = torch.where(masks > 0.1)

                # Get min/max coordinates for each batch
                x_min = x_indices.min().item()
                x_max = x_indices.max().item()
                y_min = y_indices.min().item()
                y_max = y_indices.max().item()

                # colors *= masks.unsqueeze(-1)

                colors = colors[:, y_min:y_max, x_min:x_max]
                pixels = pixels[:, y_min:y_max, x_min:x_max]
                masks = masks[:, y_min:y_max, x_min:x_max]

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]) - self.num_noise_gaussians,
                }
            )
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds  # self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(camtoworlds_all, 3)  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(camtoworlds_all, height=height)  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        elif cfg.render_traj_path == "spherical":
            camtoworlds_all = generate_spherical_poses(
                camtoworlds_all,
                lon_degree_step=10.0,
                lat_degree_step=10.0,
            )
        else:
            raise ValueError(f"Render trajectory type not supported: {cfg.render_traj_path}")

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}_{cfg.render_traj_path}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, alphas, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                require_rade=True,
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            alphas = (alphas - alphas.min()) / (alphas.max() - alphas.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")
    

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(self, camera_state: nerfview.CameraState, img_wh: tuple[int, int]):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        # # Save original opacities
        # original_opacities = self.splats["opacities"].clone()

        # # Check base colors of all Gaussians
        # if self.cfg.app_opt:
        #     colors = torch.sigmoid(self.splats["colors"])  # [N, 3]
        # else:
        #     colors = self.splats["sh0"][:, 0, :]  # [N, 3] - using DC term

        # # Identify blue Gaussians directly
        # blue_threshold = 0.6
        # is_blue = (colors[:, 2] > blue_threshold) & (colors[:, 2] > colors[:, 0] * 1.5) & (colors[:, 2] > colors[:, 1] * 1.5)

        # # Temporarily modify opacities of blue Gaussians
        # if torch.any(is_blue):
        #     self.splats["opacities"][is_blue] = float('-inf')

        # # Render with filtered Gaussians
        # render_colors, render_alphas, _ = self.rasterize_splats(
        #     camtoworlds=c2w[None],
        #     Ks=K[None],
        #     width=W,
        #     height=H,
        #     sh_degree=self.cfg.sh_degree,
        # )

        # # Restore original opacities
        # self.splats["opacities"].data.copy_(original_opacities)

        render_colors, render_alphas, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            # radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        # bkgd = torch.tensor([[0, 0, 1]], device=self.device)  # RGB: (135,206,235)
        # render_colors = render_colors + bkgd * (1.0 - render_alphas)
        # render_colors = render_colors * render_alphas + torch.ones_like(render_colors) * (1 - render_alphas)
        return render_colors[0].cpu().numpy()


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    """Main function for gsplat."""
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [torch.load(file, map_location=runner.device, weights_only=True) for file in cfg.ckpt]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        runner.save_for_blender(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            pass
        except ImportError as err:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            ) from err

    cli(main, cfg, verbose=True)
