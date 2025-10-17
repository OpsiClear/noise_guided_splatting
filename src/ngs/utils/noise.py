"""Noise utils for adding noise to the splats."""

from collections.abc import Callable

import math
import numpy as np
import open3d as o3d
import torch
import trimesh
from torch import Tensor

from ngs.utils.utils import rgb_to_sh
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_indices_in_range,
)

@torch.no_grad()
def compute_convex_hull_mesh(
    surface_points: torch.Tensor, padding: float = 0.05, scale_factor: float = 1.0
) -> tuple[trimesh.Trimesh, torch.Tensor, torch.Tensor]:
    """Compute the convex hull of surface points once.

    Args:
        surface_points: Tensor of points to compute convex hull from
        padding: Padding to add around the convex hull (as fraction of bounds size)
        scale_factor: Scale factor to apply to the convex hull

    Returns:
        Tuple of (convex hull mesh, min bounds, max bounds)
    """
    # Convert to numpy for convex hull computation
    points = surface_points.detach().cpu().numpy()

    # Create Open3D point cloud and compute convex hull
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    hull_mesh, _ = pcd.compute_convex_hull()

    # Convert to trimesh for more accurate inside/outside testing
    vertices = np.asarray(hull_mesh.vertices)
    triangles = np.asarray(hull_mesh.triangles)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    # Ensure mesh is consistently oriented
    mesh.fix_normals()
    if scale_factor != 1.0:
        centroid = mesh.centroid

        # Create a scaling matrix
        scale_matrix = np.eye(4)
        scale_matrix[:3, :3] *= scale_factor

        # Apply scaling from centroid
        mesh.apply_translation(-centroid)
        mesh.apply_transform(scale_matrix)
        mesh.apply_translation(centroid)

    # Get bounds with padding
    bounds_min = torch.tensor(mesh.bounds[0], device=surface_points.device)
    bounds_max = torch.tensor(mesh.bounds[1], device=surface_points.device)

    # Add padding
    bounds_size = bounds_max - bounds_min
    bounds_min = bounds_min - padding * bounds_size
    bounds_max = bounds_max + padding * bounds_size

    return mesh, bounds_min, bounds_max


@torch.no_grad()
def voxelize_convex_hull(
    mesh: trimesh.Trimesh, min_bound: torch.Tensor, max_bound: torch.Tensor, resolution: int, device: str
) -> torch.Tensor:
    """Voxelize the convex hull mesh at the specified resolution.

    Args:
        mesh: Convex hull mesh
        max_bound: Maximum bounds
        min_bound: Minimum bounds
        resolution: Voxel grid resolution
        device: Device to put tensor on

    Returns:
        Boolean occupancy grid tensor, bounds
    """
    # Create voxel grid
    # Find the largest dimension to ensure cubic voxels
    max_bound = max_bound.detach().cpu().numpy()
    min_bound = min_bound.detach().cpu().numpy()
    sizes = max_bound - min_bound
    max_size = sizes.max()

    # Center the grid on the points
    center = (max_bound + min_bound) / 2

    # Create a regular grid with cubic voxels
    half_size = max_size / 2
    new_min_bound = center - half_size
    new_max_bound = center + half_size
    bounds = torch.Tensor(np.array([new_min_bound, new_max_bound])).to(device)
    # Create regular grid with equal spacing in all dimensions
    grid_spacing = max_size / (resolution - 1)
    x = np.arange(resolution) * grid_spacing + new_min_bound[0]
    y = np.arange(resolution) * grid_spacing + new_min_bound[1]
    z = np.arange(resolution) * grid_spacing + new_min_bound[2]

    # Use indexing='ij' to preserve axis order
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    grid_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    # Test which points are inside the mesh
    inside = mesh.contains(grid_points)
    occupancy_grid = inside.reshape(resolution, resolution, resolution)
    occupancy_grid = torch.from_numpy(occupancy_grid).to(device)

    return occupancy_grid, bounds


@torch.no_grad()
def process_convex_hull(means: torch.nn.Parameter, resolution):
    """Create a voxelized convex hull from Gaussian means."""
    device = means.device
    mesh, bounds_min, bounds_max = compute_convex_hull_mesh(means)
    occupancy_grid, bounds = voxelize_convex_hull(mesh, bounds_min, bounds_max, resolution, device)

    return occupancy_grid, bounds


@torch.no_grad()
def add_noise_splats(
    splats: torch.nn.ParameterDict,
    occupancy_volume: torch.Tensor,
    occupancy_bounds: tuple[torch.Tensor, torch.Tensor],
    optimizers: dict[str, torch.optim.Optimizer] | None = None,
    color_init: torch.Tensor | None = None,
    init_opacity: float = 1.0,
    device: str = "cuda",
    scale_factor: float = 1.0,
) -> tuple[torch.nn.ParameterDict, dict[str, torch.optim.Optimizer] | None, int, torch.Tensor]:
    """Add fixed splats from occupancy grid points to existing splats.

    Args:
        splats: Existing splat parameters
        occupancy_volume: Occupancy volume tensor
        occupancy_bounds: Bounds of the occupancy volume
        optimizers: Optional existing optimizers to update. If None, no optimizers are returned
        color_init: Initial color value for all fixed gaussians
        init_opacity: Initial opacity value
        device: Device to put tensors on
        scale_factor: Scale factor for the gaussians

    Returns:
        Tuple of (updated ParameterDict, updated optimizers if provided else None, num_noise_gaussians, noise_gaussian_data)
    """
    occupancy_volume = occupancy_volume.to(device)
    min_bound, max_bound = occupancy_bounds
    min_bound = min_bound.to(device).float()
    max_bound = max_bound.to(device).float()

    occupied_indices = torch.nonzero(occupancy_volume).to(device)
    grid_size = torch.tensor(occupancy_volume.shape, dtype=torch.float32, device=device)

    # Convert to normalized coordinates and scale to scene bounds
    occ_points = occupied_indices.float() / (grid_size - 1)
    occ_points = occ_points * (max_bound - min_bound) + min_bound

    num_noise_gaussians = len(occ_points)

    # Create the tracking tensor for noise gaussians: [resolution, x, y, z]
    noise_gaussian_data = torch.zeros((num_noise_gaussians, 4), dtype=torch.int32, device=device)
    noise_gaussian_data[:, 0] = grid_size[0]  # Store resolution in first column
    noise_gaussian_data[:, 1:] = occupied_indices

    # Calculate voxel size directly from grid dimensions and bounds
    voxel_size = (max_bound - min_bound) / (grid_size - 1)
    # Use the smallest dimension as the base scale to ensure coverage
    min_voxel_size = torch.min(voxel_size)

    # Scale factor to convert to appropriate gaussian scale (since scales are stored in log space)
    log_scale = torch.log(scale_factor * min_voxel_size)

    # Apply scale uniformly or vary slightly for more natural appearance
    fixed_scales = log_scale.item() * torch.ones((num_noise_gaussians, 3), device=device)

    # Initialize rotations to identity quaternions
    fixed_quats = torch.zeros((num_noise_gaussians, 4), device=device)
    fixed_quats[:, 0] = 1.0  # w = 1, x,y,z = 0 for identity quaternion

    # Initialize opacities - initialize as very opaque
    eps = torch.finfo(torch.float32).eps
    fixed_opacities = torch.logit(
        torch.clamp(torch.ones((num_noise_gaussians,), device=device) * init_opacity, max=1.0 - eps, min=eps)
    )

    # Initialize colors with random RGB values if not provided
    if color_init is None:
        # Define 6 fixed RGB color tuples for better visualization
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

        # Randomly choose one of the 6 colors for each gaussian
        indices = torch.randint(0, 6, (num_noise_gaussians,), device=device)
        fixed_rgbs = color_choices[indices]
    else:
        fixed_rgbs = color_init.to(device)

    if optimizers is None:
        # If no optimizers provided, just create new parameters and return
        new_params = {}
        for name, param in splats.items():
            if name == "means":
                new_param = torch.cat([occ_points, param], dim=0)
            elif name == "scales":
                new_param = torch.cat([fixed_scales, param], dim=0)
            elif name == "quats":
                new_param = torch.cat([fixed_quats, param], dim=0)
            elif name == "opacities":
                new_param = torch.cat([fixed_opacities, param], dim=0)
            elif name == "sh0":
                sh0_channels = param.shape[1]
                fixed_colors = torch.zeros((num_noise_gaussians, sh0_channels + param.shape[1], 3), device=device)
                fixed_colors[:, 0, :] = rgb_to_sh(fixed_rgbs)
                new_param = torch.cat([fixed_colors[:, :sh0_channels, :], param], dim=0)
            elif name == "shN":
                sh0_channels = splats["sh0"].shape[1]
                shN_channels = param.shape[1]
                fixed_colors = torch.zeros((num_noise_gaussians, sh0_channels + shN_channels, 3), device=device)
                fixed_colors[:, 0, :] = rgb_to_sh(fixed_rgbs)
                new_param = torch.cat([fixed_colors[:, sh0_channels:, :], param], dim=0)
            elif name == "features":
                feature_dim = param.shape[1]
                fixed_features = torch.rand(num_noise_gaussians, feature_dim, device=device)
                new_param = torch.cat([fixed_features, param], dim=0)
            elif name == "colors":
                fixed_colors = torch.logit(fixed_rgbs)
                new_param = torch.cat([fixed_colors, param], dim=0)
            new_params[name] = torch.nn.Parameter(new_param, requires_grad=param.requires_grad)
        return torch.nn.ParameterDict(new_params), None, num_noise_gaussians, noise_gaussian_data

    # Define parameter update function
    def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
        if name == "means":
            new_param = torch.cat([occ_points, p], dim=0)
        elif name == "scales":
            new_param = torch.cat([fixed_scales, p], dim=0)
        elif name == "quats":
            new_param = torch.cat([fixed_quats, p], dim=0)
        elif name == "opacities":
            new_param = torch.cat([fixed_opacities, p], dim=0)
        elif name == "sh0":
            sh0_channels = p.shape[1]
            fixed_colors = torch.zeros((num_noise_gaussians, sh0_channels + splats["shN"].shape[1], 3), device=device)
            fixed_colors[:, 0, :] = rgb_to_sh(fixed_rgbs)
            new_param = torch.cat([fixed_colors[:, :sh0_channels, :], p], dim=0)
        elif name == "shN":
            sh0_channels = splats["sh0"].shape[1]
            shN_channels = p.shape[1]
            fixed_colors = torch.zeros((num_noise_gaussians, sh0_channels + shN_channels, 3), device=device)
            fixed_colors[:, 0, :] = rgb_to_sh(fixed_rgbs)
            new_param = torch.cat([fixed_colors[:, sh0_channels:, :], p], dim=0)
        elif name == "features":
            feature_dim = p.shape[1]
            fixed_features = torch.rand(num_noise_gaussians, feature_dim, device=device)
            new_param = torch.cat([fixed_features, p], dim=0)
        elif name == "colors":
            fixed_colors = torch.logit(fixed_rgbs)
            new_param = torch.cat([fixed_colors, p], dim=0)
        return torch.nn.Parameter(new_param, requires_grad=p.requires_grad)

    # Define optimizer state update function
    def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
        # Create zero tensor for new noise gaussians
        v_new = torch.zeros((num_noise_gaussians, *v.shape[1:]), device=device)
        # Add new states at beginning
        return torch.cat([v_new, v])

    # Update parameters and optimizer states
    _update_param_with_optimizer(param_fn, optimizer_fn, splats, optimizers)

    return splats, optimizers, num_noise_gaussians, noise_gaussian_data


@torch.no_grad()
def add_noise_splats_from_surface(
    splats: torch.nn.ParameterDict,
    optimizers: dict[str, torch.optim.Optimizer] | None = None,
    color_init: torch.Tensor | None = None,
    init_opacity: float = 1.0,
    device: str = "cuda",
) -> tuple[torch.nn.ParameterDict, dict[str, torch.optim.Optimizer] | None, int, torch.Tensor]:
    """Add fixed splats by duplicating existing splats with random colors.

    Args:
        splats: Existing splat parameters
        optimizers: Optional existing optimizers to update. If None, no optimizers are returned
        color_init: Initial color value for all fixed gaussians
        init_opacity: Initial opacity value
        device: Device to put tensors on

    Returns:
        Tuple of (updated ParameterDict, updated optimizers if provided else None, num_noise_gaussians, noise_gaussian_data)
    """
    # Use existing means and scales
    num_noise_gaussians = len(splats["means"])

    # Create a placeholder tracking tensor for noise gaussians
    # Not filling with meaningful data since we're just duplicating
    noise_gaussian_data = torch.zeros((num_noise_gaussians, 4), dtype=torch.int32, device=device)

    # Copy positions and scales from original gaussians
    fixed_means = splats["means"].clone()
    fixed_scales = splats["scales"].clone()

    # Initialize rotations to identity quaternions
    fixed_quats = torch.zeros((num_noise_gaussians, 4), device=device)
    fixed_quats[:, 0] = 1.0  # w = 1, x,y,z = 0 for identity quaternion

    # Initialize opacities
    eps = torch.finfo(torch.float32).eps
    fixed_opacities = torch.logit(
        torch.clamp(torch.ones((num_noise_gaussians,), device=device) * init_opacity, max=1.0 - eps, min=eps)
    )

    # Initialize colors with random RGB values if not provided
    if color_init is None:
        # Define 6 fixed RGB color tuples for better visualization
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

        # Randomly choose one of the 6 colors for each gaussian
        indices = torch.randint(0, 6, (num_noise_gaussians,), device=device)
        fixed_rgbs = color_choices[indices]
    else:
        fixed_rgbs = color_init.to(device)

    # Create new parameters
    new_params = {}
    new_optimizers = {} if optimizers is not None else None

    # Helper function to create new optimizer
    def create_new_optimizer(param, name):
        if optimizers is None:
            return None
        optimizer_class = type(optimizers[name])  # Get same optimizer type as original
        old_optim = optimizers[name]
        # Get the learning rate and other params from the existing optimizer
        param_group = old_optim.param_groups[0]

        return optimizer_class(
            [{"params": param, "lr": param_group["lr"], "name": name}],
            eps=old_optim.defaults["eps"],
            betas=old_optim.defaults["betas"],
        )

    # Means (positions)
    new_params["means"] = torch.nn.Parameter(torch.cat([fixed_means, splats["means"]], dim=0))
    if new_optimizers is not None:
        new_optimizers["means"] = create_new_optimizer(new_params["means"], "means")

    # Scales
    new_params["scales"] = torch.nn.Parameter(torch.cat([fixed_scales, splats["scales"]], dim=0))
    if new_optimizers is not None:
        new_optimizers["scales"] = create_new_optimizer(new_params["scales"], "scales")

    # Quaternions
    new_params["quats"] = torch.nn.Parameter(torch.cat([fixed_quats, splats["quats"]], dim=0))
    if new_optimizers is not None:
        new_optimizers["quats"] = create_new_optimizer(new_params["quats"], "quats")

    # Opacities
    new_params["opacities"] = torch.nn.Parameter(torch.cat([fixed_opacities, splats["opacities"]], dim=0))
    if new_optimizers is not None:
        new_optimizers["opacities"] = create_new_optimizer(new_params["opacities"], "opacities")

    # Colors - handle both SH and feature cases
    if "sh0" in splats:
        # SH coefficients case - infer dimensions from existing splats
        sh0_channels = splats["sh0"].shape[1]  # Get number of SH0 channels
        shN_channels = splats["shN"].shape[1]  # Get number of remaining SH channels

        fixed_colors = torch.zeros((num_noise_gaussians, sh0_channels + shN_channels, 3), device=device)
        fixed_colors[:, 0, :] = rgb_to_sh(fixed_rgbs)
        new_params["sh0"] = torch.nn.Parameter(torch.cat([fixed_colors[:, :sh0_channels, :], splats["sh0"]], dim=0))
        new_params["shN"] = torch.nn.Parameter(torch.cat([fixed_colors[:, sh0_channels:, :], splats["shN"]], dim=0))
        if new_optimizers is not None:
            new_optimizers["sh0"] = create_new_optimizer(new_params["sh0"], "sh0")
            new_optimizers["shN"] = create_new_optimizer(new_params["shN"], "shN")
    else:
        # Feature embedding case
        feature_dim = splats["features"].shape[1]
        fixed_features = torch.rand(num_noise_gaussians, feature_dim, device=device)
        fixed_colors = torch.logit(fixed_rgbs)
        new_params["features"] = torch.nn.Parameter(torch.cat([fixed_features, splats["features"]], dim=0))
        new_params["colors"] = torch.nn.Parameter(torch.cat([fixed_colors, splats["colors"]], dim=0))
        if new_optimizers is not None:
            new_optimizers["features"] = create_new_optimizer(new_params["features"], "features")
            new_optimizers["colors"] = create_new_optimizer(new_params["colors"], "colors")

    return torch.nn.ParameterDict(new_params), new_optimizers, num_noise_gaussians, noise_gaussian_data


@torch.no_grad()
def add_noise_splats_and_process_convex_hull(
    splats: torch.nn.ParameterDict,
    optimizers: dict[str, torch.optim.Optimizer],
    resolution: int,
    device: str,
    init_opacity: float,
    scale_factor: float,
    opacity_threshold: float,
    opaque_surface: bool,
) -> tuple[torch.nn.ParameterDict, dict[str, torch.optim.Optimizer], int, torch.Tensor, tuple]:
    """Add noise gaussians and process convex hull.

    Args:
        splats: Existing splat parameters
        optimizers: Existing optimizers to update
        resolution: Resolution for voxel grid
        device: Device to put tensors on
        init_opacity: Initial opacity value
        scale_factor: Scale factor for gaussian scales
        opacity_threshold: Threshold to identify surface points
        opaque_surface: Whether to set opacities to maximum for surface reconstruction
    Returns:
        Tuple of (updated ParameterDict, updated optimizers, num_noise_gaussians,
                 noise_gaussian_data, convex_hull_data)
    """
    # Identify surface points using opacity threshold
    opacity_mask = torch.sigmoid(splats["opacities"].data) > opacity_threshold
    if opaque_surface:
        # Set opacities to maximum for surface reconstruction
        eps = torch.finfo(torch.float32).eps
        splats["opacities"].data[opacity_mask] = torch.logit(
            torch.ones_like(splats["opacities"].data[opacity_mask]) * (1.0 - eps)
        )
    convex_hull_data = compute_convex_hull_mesh(splats["means"].data[opacity_mask])
    occupancy_volume, bounds = voxelize_convex_hull(
        convex_hull_data[0], convex_hull_data[1], convex_hull_data[2], resolution, device
    )

    # Add noise gaussians
    splats, optimizers, num_noise_gaussians, noise_gaussian_data = add_noise_splats(
        splats,
        occupancy_volume,
        bounds,
        optimizers,
        device=device,
        init_opacity=init_opacity,
        scale_factor=scale_factor,
    )
    return splats, optimizers, num_noise_gaussians, noise_gaussian_data, convex_hull_data


@torch.no_grad()
def add_noise_splats_coarse_to_fine(
    splats: torch.nn.ParameterDict,
    optimizers: dict[str, torch.optim.Optimizer],
    resolution: int,
    device: str,
    init_opacity: float,
    scale_factor: float,
    opacity_threshold: float,
    noise_gaussian_data: torch.Tensor | None = None,
    convex_hull_data: tuple | None = None,
) -> tuple[torch.nn.ParameterDict, dict[str, torch.optim.Optimizer], int, torch.Tensor, tuple]:
    """Add noise gaussians from convex hull with coarse-to-fine strategy.

    Args:
        splats: ParameterDict containing gaussian parameters
        optimizers: Dictionary of optimizers
        resolution: Current resolution for occupancy grid
        device: Device to put tensors on
        init_opacity: Initial opacity value for noise gaussians
        scale_factor: Scale factor for noise gaussian scales
        opacity_threshold: Threshold to identify surface points
        noise_gaussian_data: Tensor of noise gaussian data
        convex_hull_data: Optional tuple of (convex_hull_mesh, bounds_min, bounds_max) from previous stage

    Returns:
        Tuple of (updated ParameterDict, updated optimizers, number of new fixed gaussians,
                 updated noise_gaussian_data, convex_hull_data)
    """
    # If we don't have convex hull data yet, compute it
    if convex_hull_data is None:
        # Identify surface points using opacity threshold
        opacity_mask = torch.sigmoid(splats["opacities"].data) > opacity_threshold
        surface_points = splats["means"].data[opacity_mask]

        # Compute convex hull (this is done only once)
        convex_hull_mesh, bounds_min, bounds_max = compute_convex_hull_mesh(surface_points)
        convex_hull_data = (convex_hull_mesh, bounds_min, bounds_max)
    else:
        # Unpack existing convex hull data
        convex_hull_mesh, bounds_min, bounds_max = convex_hull_data

    # Voxelize the convex hull at the current resolution
    occupancy_volume, bounds = voxelize_convex_hull(convex_hull_mesh, bounds_min, bounds_max, resolution, device)

    # Ensure bounds are on the correct device and type
    bounds_min, bounds_max = bounds
    bounds_min = bounds_min.to(device).float()
    bounds_max = bounds_max.to(device).float()

    # Create a mask to track which voxels already have gaussians
    voxels_with_gaussians = torch.zeros_like(occupancy_volume, dtype=torch.bool)

    # If we have existing noise gaussians, mark their corresponding voxels in the new grid
    if noise_gaussian_data is not None and len(noise_gaussian_data) > 0:
        # Get all relevant data at once
        orig_resolutions = noise_gaussian_data[:, 0]
        grid_coords = noise_gaussian_data[:, 1:].long()

        # Vectorized approach for power-of-2 grid resolutions
        # For each gaussian, we need to calculate:
        # 1. The scale factor between original resolution and current resolution
        # 2. The base coordinate in the new grid
        # 3. All the voxels that need to be marked as occupied

        # Calculate scaling factors for each gaussian (how many times larger the new grid is)
        scale_factors = resolution / orig_resolutions

        # Compute the base coordinates in the new grid
        base_coords = (grid_coords.float() * scale_factors[:, None]).long()

        # Create a list to accumulate all coordinates that need to be marked
        all_fine_coords = []

        # Process each unique scale factor separately
        for unique_scale in torch.unique(scale_factors):
            scale_int = int(unique_scale.item())
            mask = scale_factors == unique_scale

            # Get the base coordinates for gaussians with this scale factor
            curr_base_coords = base_coords[mask]

            if len(curr_base_coords) == 0:
                continue

            # Create offset combinations based on the scale factor
            # For scale=2, we need offsets (0,0,0), (0,0,1), ..., (1,1,1)
            # For scale=4, we need offsets from (0,0,0) to (3,3,3)
            offsets = torch.tensor(
                [[x, y, z] for x in range(scale_int) for y in range(scale_int) for z in range(scale_int)], device=device
            )

            # Apply offsets to base coordinates
            curr_fine_coords = curr_base_coords.unsqueeze(1) + offsets.unsqueeze(0)

            # Reshape to [num_gaussians_at_this_scale * scale^3, 3]
            curr_fine_coords = curr_fine_coords.reshape(-1, 3)

            # Add to our collection
            all_fine_coords.append(curr_fine_coords)

        # Combine all coordinates
        if all_fine_coords:
            fine_coords = torch.cat(all_fine_coords, dim=0)

            # Filter out coordinates outside the grid bounds
            valid_mask = ((fine_coords >= 0) & (fine_coords < resolution)).all(dim=1)
            valid_coords = fine_coords[valid_mask]

            # Mark all valid positions in the grid at once
            if len(valid_coords) > 0:
                # Use unique to handle potential duplicates
                unique_coords = torch.unique(valid_coords, dim=0)
                voxels_with_gaussians[unique_coords[:, 0], unique_coords[:, 1], unique_coords[:, 2]] = True

    # Find voxels that are occupied AND don't already have gaussians
    available_voxels = occupancy_volume & ~voxels_with_gaussians
    occ_indices = torch.nonzero(available_voxels)  # [N, 3] tensor of indices
    num_new_gaussians = len(occ_indices)

    if num_new_gaussians == 0:
        return splats, optimizers, 0, noise_gaussian_data, convex_hull_data

    # Get the dimensions of the occupancy volume
    voxel_dims = torch.tensor(occupancy_volume.shape, dtype=torch.float32, device=device)

    # Convert indices to normalized coordinates and scale to scene bounds
    occ_points = occ_indices.float() / (voxel_dims - 1)
    occ_points = occ_points * (bounds_max - bounds_min) + bounds_min

    # Calculate voxel size directly from grid dimensions and bounds
    voxel_size = (bounds_max - bounds_min) / (voxel_dims - 1)
    min_voxel_size = torch.min(voxel_size)

    # Scale factor to convert to appropriate gaussian scale (since scales are stored in log space)
    log_scale = torch.log(scale_factor * min_voxel_size)

    # Apply scale uniformly
    scale_init = log_scale.item() * torch.ones((num_new_gaussians, 3), device=device)

    # Initialize colors for new gaussians
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
    indices = torch.randint(0, 6, (num_new_gaussians,), device=device)
    colors_init = color_choices[indices]

    # Create new tracking data for the new noise gaussians
    new_noise_data = torch.zeros((num_new_gaussians, 4), dtype=torch.int32, device=device)
    new_noise_data[:, 0] = resolution  # Store current resolution
    new_noise_data[:, 1:] = occ_indices  # Store grid coordinates

    # Combine with existing noise gaussian data if it exists
    if noise_gaussian_data is not None and len(noise_gaussian_data) > 0:
        combined_noise_data = torch.cat([new_noise_data, noise_gaussian_data], dim=0)
    else:
        combined_noise_data = new_noise_data

    # Define parameter update function
    def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
        if name == "means":
            new_param = torch.cat([occ_points, p.clone()], dim=0)
        elif name == "scales":
            new_param = torch.cat([scale_init, p.clone()], dim=0)
        elif name == "quats":
            quat_init = torch.zeros((num_new_gaussians, 4), device=device)
            quat_init[:, 0] = 1.0  # identity quaternion
            new_param = torch.cat([quat_init, p.clone()], dim=0)
        elif name == "opacities":
            eps = torch.finfo(torch.float32).eps
            opacity_init = torch.logit(
                torch.clamp(torch.ones((num_new_gaussians,), device=device) * init_opacity, max=1.0 - eps, min=0.005)
            )
            new_param = torch.cat([opacity_init, p.clone()], dim=0)
        elif name == "sh0":
            new_param = torch.cat([rgb_to_sh(colors_init).unsqueeze(1), p.clone()], dim=0)
        elif name == "shN":
            sh_shape = p.shape[1:]
            sh_init = torch.zeros((num_new_gaussians,) + sh_shape, device=device)
            new_param = torch.cat([sh_init, p.clone()], dim=0)
        elif name == "colors":
            new_param = torch.cat([torch.logit(colors_init), p.clone()], dim=0)
        elif name == "features":
            feature_shape = p.shape[1:]
            feature_init = torch.zeros((num_new_gaussians,) + feature_shape, device=device)
            new_param = torch.cat([feature_init, p.clone()], dim=0)
        else:
            shape = p.shape[1:]
            init = torch.zeros((num_new_gaussians,) + shape, device=device)
            new_param = torch.cat([init, p.clone()], dim=0)
        return torch.nn.Parameter(new_param, requires_grad=p.requires_grad)

    # Define optimizer state update function
    def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
        # Create zero tensor for new noise gaussians
        v_new = torch.zeros((num_new_gaussians, *v.shape[1:]), device=device)
        # Add new states at beginning
        return torch.cat([v_new, v])

    # Update parameters and optimizer states
    _update_param_with_optimizer(param_fn, optimizer_fn, splats, optimizers)

    return splats, optimizers, num_new_gaussians, combined_noise_data, convex_hull_data


@torch.no_grad()
def remove_low_opacity_splats(
    splats: torch.nn.ParameterDict,
    optimizers: dict[str, torch.optim.Optimizer],
    opacity_threshold: float = 0.01,
    device: str = "cuda",
) -> tuple[torch.nn.ParameterDict, dict[str, torch.optim.Optimizer], int]:
    """Remove gaussians with opacity below threshold from parameter dictionary.

    Args:
        splats: ParameterDict containing gaussian parameters
        optimizers: Dictionary of optimizers
        opacity_threshold: Threshold below which gaussians will be removed
        device: Device to put tensors on

    Returns:
        Tuple of (updated ParameterDict, updated optimizers, number of gaussians removed)
    """
    # Get mask of gaussians to keep
    opacities = torch.sigmoid(splats["opacities"].data)
    keep_mask = opacities >= opacity_threshold
    num_removed = (~keep_mask).sum().item()

    # Create new parameter dictionary without low opacity splats
    new_params = {}
    new_optimizers = {}

    # Helper function to create new optimizer
    def create_new_optimizer(param, name):
        optimizer_class = type(optimizers[name])
        old_optim = optimizers[name]
        param_group = old_optim.param_groups[0]
        return optimizer_class(
            [{"params": param, "lr": param_group["lr"], "name": name}],
            eps=old_optim.defaults["eps"],
            betas=old_optim.defaults["betas"],
        )

    # Filter each parameter tensor
    for name, param in splats.items():
        new_params[name] = torch.nn.Parameter(param.data[keep_mask].clone())
        new_optimizers[name] = create_new_optimizer(new_params[name], name)

    return torch.nn.ParameterDict(new_params), new_optimizers, num_removed


@torch.no_grad()
def remove_noise_splats(
    splats: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
    optimizers: dict[str, torch.optim.Optimizer],
    num_noise_gaussians: int,
    state: dict[str, Tensor] = None,
    device: str = "cuda",
) -> None:
    """Remove fixed noise Gaussians from the parameters.

    Args:
        splats: A dictionary of parameters.
        optimizers: A dictionary of optimizers.
        num_noise_gaussians: Number of fixed gaussians to remove.
        state: Additional state dictionary to update. Default: None.
        device: Device to use. Default: "cuda".
    """
    if num_noise_gaussians == 0:
        return

    # Create a mask where False = keep (non-noise), True = remove (noise)
    mask = torch.zeros(next(iter(splats.values())).shape[0], dtype=torch.bool, device=device)
    mask[:num_noise_gaussians] = True

    def param_fn(name: str, p: Tensor) -> Tensor:
        # Keep only non-noise Gaussians (where mask is False)
        return torch.nn.Parameter(p[~mask], requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        # Update optimizer state to match new parameters
        return v[~mask]

    # Call helper to update parameters and optimizer states
    _update_param_with_optimizer(param_fn, optimizer_fn, splats, optimizers)

    # Update any additional state tensors
    if state is not None:
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v[~mask]


def _update_param_with_optimizer(
    param_fn: Callable[[str, Tensor], Tensor],
    optimizer_fn: Callable[[str, Tensor], Tensor],
    params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
    optimizers: dict[str, torch.optim.Optimizer],
    names: list[str] | None = None,
):
    """Update the parameters and the state in the optimizers with defined functions.

    Args:
        param_fn: A function that takes the name of the parameter and the parameter itself,
            and returns the new parameter.
        optimizer_fn: A function that takes the key of the optimizer state and the state value,
            and returns the new state value.
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        names: A list of key names to update. If None, update all. Default: None.
    """
    if names is None:
        # If names is not provided, update all parameters
        names = list(params.keys())

    for name in names:
        param = params[name]
        new_param = param_fn(name, param)
        params[name] = new_param
        if name not in optimizers:
            if param.requires_grad:
                raise ValueError(
                    f"Optimizer for {name} is not found, but the parameter is trainable. "
                    f"Got requires_grad={param.requires_grad}"
                )
            continue
        optimizer = optimizers[name]
        for i in range(len(optimizer.param_groups)):
            param_state = optimizer.state[param]
            del optimizer.state[param]
            for key in param_state.keys():
                if key != "step":
                    v = param_state[key]
                    param_state[key] = optimizer_fn(key, v)
            optimizer.param_groups[i]["params"] = [new_param]
            optimizer.state[new_param] = param_state


@torch.no_grad()
def reset_color_noise_splats(
    splats: torch.nn.ParameterDict,
    optimizers: dict[str, torch.optim.Optimizer],
    num_noise_gaussians: int,
    device: str = "cuda",
    noise_color: tuple[float, float, float] | None = None,
) -> None:
    """Reset the color of the noise splats to the original color.

    Args:
        splats: ParameterDict containing the splat parameters
        optimizers: Dictionary of optimizers
        num_noise_gaussians: Number of fixed gaussians to reset
        device: Device to put tensors on
        noise_color: Color to set the noise splats to. If None, a random color will be chosen.
    """
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

    if noise_color is None:
        # Randomly choose one of the 6 colors for each gaussian
        indices = torch.randint(0, 6, (num_noise_gaussians,), device=device)
        new_rgbs = color_choices[indices]
    else:
        new_rgbs = torch.ones((num_noise_gaussians, 3), device=device) * torch.tensor(noise_color, device=device)

    if "sh0" in splats:
        # SH coefficients case
        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            if name == "sh0":
                p_new = p.clone()
                p_new[:num_noise_gaussians, 0, :] = rgb_to_sh(new_rgbs)
                return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
            elif name == "shN":
                p_new = p.clone()
                p_new[:num_noise_gaussians] = 0.0
                return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
            return p

        # Only modify optimizer state for the first num_noise_gaussians elements
        def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            v_new = v.clone()
            v_new[:num_noise_gaussians] = 0
            return v_new

        _update_param_with_optimizer(param_fn, optimizer_fn, splats, optimizers, names=["sh0", "shN"])
    else:
        # Feature embedding case
        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            if name == "colors":
                p_new = p.clone()
                p_new[:num_noise_gaussians] = torch.logit(new_rgbs)
                return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
            return p

        def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            v_new = v.clone()
            v_new[:num_noise_gaussians] = 0
            return v_new

        _update_param_with_optimizer(param_fn, optimizer_fn, splats, optimizers, names=["colors"])


@torch.no_grad()
def reset_opacity_noise_splats(
    splats: torch.nn.ParameterDict,
    optimizers: dict[str, torch.optim.Optimizer],
    num_noise_gaussians: int,
    opacity: float,
    device: str = "cuda",
) -> None:
    """Reset the opacity of the noise splats to the original opacity.

    Args:
        splats: ParameterDict containing the splat parameters
        optimizers: Dictionary of optimizers
        num_noise_gaussians: Number of fixed gaussians to reset
        opacity: Opacity value to set
        device: Device to put tensors on
    """
    eps = torch.finfo(torch.float32).eps
    new_opacities = torch.logit(torch.clamp(torch.ones((num_noise_gaussians,), device=device) * opacity, max=1.0 - eps))

    def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
        p_new = p.clone()
        p_new[:num_noise_gaussians] = new_opacities
        return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
        v_new = v.clone()
        v_new[:num_noise_gaussians] = 0
        return v_new

    _update_param_with_optimizer(param_fn, optimizer_fn, splats, optimizers, names=["opacities"])


@torch.no_grad()
def identify_noise_gaussians_to_prune(
    means, 
    quats, 
    scales, 
    opacities, 
    viewmats, 
    Ks, 
    width, 
    height, 
    num_noise_gaussians,
    near_plane=0.01, 
    far_plane=1e10,  
    radius_clip=0.0, 
    eps2d=0.3,
    tile_size=16,
    scale_multiplier=3.0,  # Multiplier to determine depth threshold from noise Gaussian scale
    downsample_factor=1.0,  # Factor to downsample the rendering resolution
):
    """
    Returns a boolean mask indicating which noise Gaussians should be pruned because
    they appear in front of surface Gaussians or are embedded within a surface.
    
    Args:
        means: Tensor of shape [N, 3]
        quats: Tensor of shape [N, 4]
        scales: Tensor of shape [N, 3]
        opacities: Tensor of shape [N]
        viewmats: Tensor of shape [C, 4, 4]
        Ks: Tensor of shape [C, 3, 3]
        width: int
        height: int
        num_noise_gaussians: Number of noise Gaussians at the beginning of the array
        near_plane: float
        far_plane: float
        radius_clip: float
        eps2d: float
        tile_size: int
        scale_multiplier: float - Multiplier for scale to determine depth threshold
        downsample_factor: float - Factor to downsample rendering resolution for memory efficiency
    
    Returns:
        prune_mask: Boolean tensor [N] where True indicates a Gaussian that should be pruned
    """

    device = means.device
    N = means.shape[0]
    C = viewmats.shape[0]
    
    # Skip the whole process if there are no noise Gaussians
    if num_noise_gaussians == 0:
        return torch.zeros(N, dtype=torch.bool, device=device)
    
    # Apply downsampling to resolution and camera intrinsics
    if downsample_factor != 1.0:
        # Downsample width and height
        ds_width = int(width / downsample_factor)
        ds_height = int(height / downsample_factor)
        
        # Adjust camera intrinsics for downsampling
        ds_Ks = Ks.clone()
        # Scale focal length and principal point
        ds_Ks[:, 0, 0] /= downsample_factor  # fx
        ds_Ks[:, 1, 1] /= downsample_factor  # fy
        ds_Ks[:, 0, 2] /= downsample_factor  # cx
        ds_Ks[:, 1, 2] /= downsample_factor  # cy
    else:
        ds_width = width
        ds_height = height
        ds_Ks = Ks
    
    # Initialize the pruning mask
    to_prune = torch.zeros(N, dtype=torch.bool, device=device)
    
    # Only project Gaussians to 2D, without full rendering
    proj_results = fully_fused_projection(
        means,
        covars=None,
        quats=quats,
        scales=scales,
        viewmats=viewmats,
        Ks=ds_Ks,  # Use downsampled camera intrinsics
        width=ds_width,
        height=ds_height,
        eps2d=eps2d,
        packed=False,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=False,
        calc_compensations=False
    )
    
    # Extract projection results for unpacked format
    radii, means2d, depths, conics, *_ = proj_results
    
    # Identify intersecting tiles
    tile_width = math.ceil(ds_width / float(tile_size))
    tile_height = math.ceil(ds_height / float(tile_size))
    
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=False,
        n_images=C,
        image_ids=None,
        gaussian_ids=None,
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)
    
    # Get ALL gaussian-pixel interactions with their depth ordering
    transmittances = torch.ones((C, ds_height, ds_width), device=device)
    gauss_ids, pixel_ids, camera_ids = rasterize_to_indices_in_range(
        range_start=0,
        range_end=N,
        transmittances=transmittances,
        means2d=means2d,
        conics=conics,
        opacities=opacities.repeat(C, 1),
        image_width=ds_width,
        image_height=ds_height,
        tile_size=tile_size,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids
    )
    
    # Initialize mask for marking gaussians to prune
    to_prune = torch.zeros(N, dtype=torch.bool, device=device)
    
    # If no gaussians found, return empty mask
    if gauss_ids.numel() == 0:
        return to_prune
    
    # Create unique pixel identifiers and categorize Gaussians
    unique_pixels = camera_ids * ds_width * ds_height + pixel_ids
    is_noise = gauss_ids < num_noise_gaussians
    is_surface = ~is_noise
    
    # Get unique pixels and corresponding indices
    unique_pixel_values, inverse_indices = torch.unique(unique_pixels, return_inverse=True)
    num_unique_pixels = unique_pixel_values.shape[0]
    
    # Create a tensor of indices
    indices = torch.arange(gauss_ids.shape[0], device=device)
    
    # Where there's a surface Gaussian, use its index; otherwise use a large value
    surface_indices = torch.where(is_surface, indices, torch.tensor(gauss_ids.shape[0], device=device))
    
    # Find the minimum surface index for each pixel
    min_surface_indices = torch.full((num_unique_pixels,), gauss_ids.shape[0], dtype=torch.long, device=device)
    min_surface_indices.scatter_reduce_(0, inverse_indices, surface_indices, reduce="min", include_self=False)
    
    # For each position, get the first surface index for its pixel
    first_surface_idx = min_surface_indices[inverse_indices]
    
    # Mark for pruning any noise gaussian that appears before the first surface for its pixel
    to_prune_front = (indices < first_surface_idx) & is_noise
    to_prune[gauss_ids[to_prune_front]] = True
    
    
    return to_prune
