from dataclasses import dataclass, field
from typing import Literal, Optional
import tyro
from ngs.ngs_strategy import DefaultStrategy


@dataclass
class EvaluatorConfig:
    """Configuration for Gaussian Splatting.

    Attributes:
        pcloud: Path to the .ply files
        data_dir: Path to the scan containing colmap and images
        data_factor: Downsample factor for the dataset
        result_dir: Directory to save results
        test_every: Every N images there is a test image
        patch_size: Random crop size for training (experimental)
        global_scale: A global scaler that applies to the scene size related parameters
        camera_model: Camera model type
        sh_degree: Degree of spherical harmonics
        near_plane: Near plane clipping distance
        far_plane: Far plane clipping distance
        eps: Epsilon for log10
        packed: Use packed mode for rasterization (less memory, slightly slower)
        lpips_net: LPIPS network type to use
        device: Device to run the model on
    """

    surface_pcloud: str | None = None
    inside_pcloud: str | None = None
    data_dir: str | None = None
    data_factor: int = 1
    result_dir: str = "results"
    test_every: int = 8
    patch_size: Optional[int] = None
    global_scale: float = 1.0
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    batch_size: int = 1
    sh_degree: int = 3
    near_plane: float = 0.01
    far_plane: float = 10000
    eps: float = 1e-10
    packed: bool = False
    lpips_net: Literal["vgg", "alex"] = "alex"
    device: str = "cuda:0"


@dataclass
class TrainerConfig:
    """Configuration for Gaussian Splatting.

    Attributes:
        gpu_indices: GPU indices to use
        disable_viewer: Disable viewer
        ckpt: Path to the .pt files. If provided, it will skip training and run evaluation only
        compression: Name of compression strategy to use
        render_traj_path: Render trajectory path ("interp", "ellipse", "spiral")
        data_dir: Path to the Mip-NeRF 360 dataset
        data_factor: Downsample factor for the dataset
        result_dir: Directory to save results
        test_every: Every N images there is a test image
        patch_size: Random crop size for training (experimental)
        global_scale: A global scaler that applies to the scene size related parameters
        normalize_world_space: Normalize the world space
        camera_model: Camera model type
        port: Port for the viewer server
        batch_size: Batch size for training. Learning rates are scaled automatically
        steps_scaler: A global factor to scale the number of training steps
        reset_noise_interval: Interval to reset noise
        rade_step: Step to start Rade normal consistency loss
        noise_step_1: Step to add noise
        unfreeze_step_1: Step to unfreeze
        noise_step_2: Step to add noise at second stage
        unfreeze_step_2: Step to unfreeze at second stage
        opa_threshold: Opacity threshold for surface reconstruction for convex hull
        init_noise_opa: Initial opacity for noise gaussians
        opaque_surface: Whether to set opacities of surface to 1 when freezing surface
        max_steps: Number of training steps
        eval_steps: Steps to evaluate the model
        save_steps: Steps to save the model
        init_type: Initialization strategy
        init_num_pts: Initial number of GSs (ignored if using sfm)
        init_extent: Initial extent of GSs as a multiple of the camera extent (ignored if using sfm)
        sh_degree: Degree of spherical harmonics
        sh_degree_interval: Turn on another SH degree every this steps
        init_opa: Initial opacity of GS
        init_scale: Initial scale of GS
        ssim_lambda: Weight for SSIM loss
        near_plane: Near plane clipping distance
        far_plane: Far plane clipping distance
        strategy: Strategy for GS densification
        packed: Use packed mode for rasterization (less memory, slightly slower)
        sparse_grad: Use sparse gradients for optimization (experimental)
        visible_adam: Use visible adam from Taming 3DGS (experimental)
        antialiased: Anti-aliasing in rasterization (might slightly hurt quantitative metrics)
        random_bkgd: Use random background for training to discourage transparency
        opacity_reg: Opacity regularization
        scale_reg: Scale regularization
        pose_opt: Enable camera optimization
        pose_opt_lr: Learning rate for camera optimization
        pose_opt_reg: Regularization for camera optimization as weight decay
        pose_noise: Add noise to camera extrinsics (only to test camera pose optimization)
        app_opt: Enable appearance optimization (experimental)
        app_embed_dim: Appearance embedding dimension
        feature_dim: Feature embedding dimension
        app_opt_lr: Learning rate for appearance optimization
        app_opt_reg: Regularization for appearance optimization as weight decay
        use_bilateral_grid: Enable bilateral grid (experimental)
        bilateral_grid_shape: Shape of the bilateral grid (X, Y, W)
        depth_loss: Enable depth loss (experimental)
        depth_lambda: Weight for depth loss
        tb_every: Dump information to tensorboard every this steps
        tb_save_image: Save training images to tensorboard
        lpips_net: LPIPS network type to use
    """
    disable_viewer: bool = False
    ckpt: list[str] | None = None
    compression: Literal["png"] | None = None
    render_traj_path: str = "ellipse"
    data_dir: str = ""
    data_factor: int = 1
    result_dir: str = "results"
    test_every: int = 8
    patch_size: int | None = None
    global_scale: float = 1.0
    normalize_world_space: bool = False
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    port: int = 8080
    batch_size: int = 1
    steps_scaler: float = 1.0
    render_every: int = 100

    # Noise
    optimize_foreground: bool = False
    load_images_in_memory: bool = False
    hull_filter_step: int = -1
    reset_noise_interval: int = 1500
    rade_step: int = -1
    noise_steps: list[int] = field(default_factory=lambda: [6000])
    unfreeze_steps: list[int] = field(default_factory=lambda: [1000])
    opa_threshold: float = 0.0
    init_noise_opa: float = 0.1
    freeze_noise_opa: float = 1.0
    opaque_surface: bool = False
    voxel_resolution: int = 32
    erosion_steps: int = 3
    increase_erosion_steps: bool = True
    foreground_loss: bool = True
    noise_color: tuple[float, float, float] | None = None
    reset_lr: bool = True

    max_steps: int = 30_000
    eval_steps: list[int] = field(default_factory=lambda: [7000, 30000])
    save_steps: list[int] = field(default_factory=lambda: [7000, 30000])
    init_type: Literal["sfm", "mcmc"] = "sfm"
    init_num_pts: int = 100_000
    init_extent: float = 1.0
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_opa: float = 0.1
    init_scale: float = 1.0
    ssim_lambda: float = 0.2
    near_plane: float = 0.01
    far_plane: float = 10000

    packed: bool = False
    sparse_grad: bool = False
    visible_adam: bool = False
    antialiased: bool = False
    random_bkgd: bool = False
    opacity_reg: float = 0.0
    scale_reg: float = 0.0

    pose_opt: bool = False
    pose_opt_lr: float = 0.000001
    pose_opt_reg: float = 0.0000001
    pose_noise: float = 0.0

    app_opt: bool = False
    app_embed_dim: int = 16
    feature_dim: int = 32
    app_opt_lr: float = 0.001
    app_opt_reg: float = 0.000001

    use_bilateral_grid: bool = False
    bilateral_grid_shape: tuple[int, int, int] = (16, 16, 8)
    depth_loss: bool = False
    depth_lambda: float = 0.01

    tb_every: int = 100
    tb_save_image: bool = False
    lpips_net: Literal["vgg", "alex"] = "alex"

    strategy: DefaultStrategy = field(default_factory=DefaultStrategy)

    def adjust_steps(self, factor: float):
        """Adjust the steps for training."""
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
        strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
        strategy.reset_every = int(strategy.reset_every * factor)
        strategy.refine_every = int(strategy.refine_every * factor)

if __name__ == "__main__":
    # Example usage with tyro
    config = tyro.cli(EvaluatorConfig)
    print(config) 