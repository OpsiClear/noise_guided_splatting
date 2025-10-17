"""Utils module for gsplat pipeline."""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors
from torch import Tensor


class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        """Initialize camera pose optimization module.

        Args:
            n: Number of cameras to optimize.
        """
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        """Initialize the camera pose optimization module with zero embeddings."""
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        """Initialize the camera pose optimization module with random embeddings."""
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        if camtoworlds.shape[:-2] != embed_ids.shape:
            raise ValueError(f"camtoworlds.shape[:-2] ({camtoworlds.shape[:-2]}) != embed_ids.shape ({embed_ids.shape})")
        batch_shape = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(drot + self.identity.expand(*batch_shape, -1))  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        """Initialize the appearance optimization module.

        Args:
            n: Number of cameras to optimize.
            feature_dim: Dimension of the feature space.
            embed_dim: Dimension of the embedding space.
            sh_degree: Degree of the spherical harmonics.
            mlp_width: Width of the MLP.
            mlp_depth: Depth of the MLP.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width))
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)
            sh_degree: Degree of spherical harmonics to use

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        c, n = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(c, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, n, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(c, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(c, n, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """Converts 6D rotation representation by Zhou et al. [1] to rotation matrix.

    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.

    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def knn(x: Tensor, k: int = 4) -> Tensor:
    """K-nearest neighbors."""
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    """Convert RGB to SH."""
    c0 = 0.28209479177387814
    return (rgb - 0.5) / c0


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ref: https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/general_utils.py#L163
def colormap(img, cmap="jet"):
    """Colormap for images."""
    w, h = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(h / dpi, w / dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data).float().permute(2, 0, 1)
    plt.close()
    return img


def apply_float_colormap(img: torch.Tensor, colormap: str = "turbo") -> torch.Tensor:
    """Convert single channel to a color img.

    Args:
        img (torch.Tensor): (..., 1) float32 single channel image.
        colormap (str): Colormap for img.

    Returns:
        (..., 3) colored img with colors in [0, 1].
    """
    img = torch.nan_to_num(img, 0)
    if colormap == "gray":
        return img.repeat(1, 1, 3)
    img_long = (img * 255).long()
    img_long_min = torch.min(img_long)
    img_long_max = torch.max(img_long)
    if img_long_min < 0:
        raise ValueError(f"the min value is {img_long_min}")
    if img_long_max > 255:
        raise ValueError(f"the max value is {img_long_max}")
    return torch.tensor(
        colormaps[colormap].colors,  # type: ignore
        device=img.device,
    )[img_long[..., 0]]


def apply_depth_colormap(
    depth: torch.Tensor,
    acc: torch.Tensor = None,
    near_plane: float = None,
    far_plane: float = None,
) -> torch.Tensor:
    """Converts a depth image to color for easier analysis.

    Args:
        depth (torch.Tensor): (..., 1) float32 depth.
        acc (torch.Tensor | None): (..., 1) optional accumulation mask.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.

    Returns:
        (..., 3) colored depth image with colors in [0, 1].
    """
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    img = apply_float_colormap(depth, colormap="turbo")
    if acc is not None:
        img = img * acc + (1.0 - acc)
    return img


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to SH."""
    c0 = 0.28209479177387814
    return (rgb - 0.5) / c0

def save_ply(gaussian_data, filename, device="cuda"):
    """Save 3D Gaussian model data to PLY file matching the original format.

    Args:
        gaussian_data: Dictionary containing Gaussian parameters
        filename: Output PLY file path
        device: Computation device
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Extract and prepare data
    xyz = gaussian_data["means"].detach().cpu().numpy()
    normals = np.zeros_like(xyz)  # Empty normals

    # Features DC and rest need to be properly reshaped
    if "sh0" in gaussian_data:
        # SH coefficients case: [N, K, 3] -> [N, 3K]
        f_dc = gaussian_data["sh0"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = gaussian_data["shN"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    else:
        # Colors and features case: colors [N, 3], features [N, feature_dim]
        f_dc = gaussian_data["colors"].detach().cpu().numpy()  # Already [N, 3]
        f_rest = gaussian_data["features"].detach().cpu().numpy()  # Already [N, feature_dim]

    # Other properties - ensure correct dimensions
    opacities = gaussian_data["opacities"].detach().cpu().numpy()
    if opacities.ndim == 1:
        opacities = opacities.reshape(-1, 1)

    scales = gaussian_data["scales"].detach().cpu().numpy()
    rotations = gaussian_data["quats"].detach().cpu().numpy()

    # Construct attribute list
    attributes = ["x", "y", "z", "nx", "ny", "nz"]

    # Add feature DC terms
    for i in range(f_dc.shape[1]):
        attributes.append(f"f_dc_{i}")

    # Add feature rest terms
    for i in range(f_rest.shape[1]):
        attributes.append(f"f_rest_{i}")

    # Add opacity
    attributes.append("opacity")

    # Add scales
    for i in range(scales.shape[1]):
        attributes.append(f"scale_{i}")

    # Add rotations
    for i in range(rotations.shape[1]):
        attributes.append(f"rot_{i}")

    # Create dtype for structured array
    dtype_full = [(attribute, "f4") for attribute in attributes]

    # Create empty array with the defined dtype
    elements = np.empty(xyz.shape[0], dtype=dtype_full)

    # Ensure all arrays have correct shape before concatenation
    if f_dc.ndim == 1:
        f_dc = f_dc.reshape(-1, 1)
    if f_rest.ndim == 1:
        f_rest = f_rest.reshape(-1, 1)


    # Concatenate all attributes in the correct order
    attributes = np.concatenate(
        (
            xyz,  # positions [N, 3]
            normals,  # empty normals [N, 3]
            f_dc,  # DC features [N, DC]
            f_rest,  # rest features [N, REST]
            opacities,  # opacity [N, 1]
            scales,  # scales [N, 3]
            rotations,  # rotations [N, 4]
        ),
        axis=1,
    )

    # Fill the structured array
    elements[:] = list(map(tuple, attributes))

    # Create PLY element and save
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(filename)



def load_ply(ply_path, sh_degree, device="cuda"):
    """Load a PLY file and convert it to a parameter dictionary for rendering."""

    plydata = PlyData.read(ply_path)
    vertices = plydata["vertex"].data

    params_dict = {}

    # Extract positions
    xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    params_dict["means"] = torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device=device))

    # Get field names
    field_names = vertices.dtype.names

    # Count DC and rest coefficients
    dc_count = sum(1 for name in field_names if name.startswith("f_dc_"))
    rest_count = sum(1 for name in field_names if name.startswith("f_rest_"))

    if dc_count >= 3:  # At least 3 for RGB
        # Extract f_dc terms (typically 3 for RGB)
        f_dc = np.vstack([vertices[f"f_dc_{i}"] for i in range(dc_count)]).T

        k_minus_1 = (sh_degree + 1) ** 2 - 1

        # Create sh0 with shape [N, 1, 3] from f_dc
        sh0 = torch.tensor(f_dc, dtype=torch.float32, device=device).reshape(-1, 3).unsqueeze(1)
        params_dict["sh0"] = torch.nn.Parameter(sh0)

        # Handle f_rest coefficients for higher-order SH
        if rest_count > 0:
            # Extract f_rest terms (3 * (K-1) coefficients per Gaussian)
            rest_features = np.vstack([vertices[f"f_rest_{i}"] for i in range(rest_count)]).T

            # The PLY stores SH in format [N, 3*(K-1)]
            # We need to reshape to [N, K-1, 3]
            n_gaussians = xyz.shape[0]

            # Check if dimensions match what we expect
            if rest_features.shape[1] == 3 * k_minus_1:
                # Directly reshape - this fixes the issue with coefficient ordering
                # When saving: sh is transposed (1,2) and flattened, so reshape correctly
                shN = torch.tensor(rest_features, dtype=torch.float32, device=device)
                shN = shN.reshape(n_gaussians, 3, k_minus_1).transpose(1, 2)
            else:
                print(
                    f"Warning: SH coefficients shape mismatch. Using zeros. "
                    f"Expected {3 * k_minus_1} coefficients, got {rest_features.shape[1]}"
                )
                shN = torch.zeros((n_gaussians, k_minus_1, 3), dtype=torch.float32, device=device)

            params_dict["shN"] = torch.nn.Parameter(shN)
        else:
            # If no rest coefficients, create zeros with correct shape
            params_dict["shN"] = torch.nn.Parameter(
                torch.zeros((xyz.shape[0], k_minus_1, 3), dtype=torch.float32, device=device)
            )
    else:
        # Handle simple colors case (not SH coefficients)
        if "f_dc_0" in field_names:
            colors = np.vstack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]).T
        else:
            # Fallback if no color fields found
            colors = np.ones((len(xyz), 3))

        params_dict["colors"] = torch.nn.Parameter(torch.tensor(colors, dtype=torch.float32, device=device))

        # Handle features if present (for appearance module)
        if rest_count > 0:
            features = np.vstack([vertices[f"f_rest_{i}"] for i in range(rest_count)]).T
            params_dict["features"] = torch.nn.Parameter(torch.tensor(features, dtype=torch.float32, device=device))
        else:
            # Create empty features tensor
            params_dict["features"] = torch.nn.Parameter(torch.zeros((len(xyz), 0), dtype=torch.float32, device=device))

    # Extract opacities - left in sigmoid space for the parameter
    opacities = vertices["opacity"]
    params_dict["opacities"] = torch.nn.Parameter(torch.tensor(opacities, dtype=torch.float32, device=device))

    # Extract scales
    scales = np.vstack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]]).T
    params_dict["scales"] = torch.nn.Parameter(torch.tensor(scales, dtype=torch.float32, device=device))

    # Extract rotations (quaternions)
    rotations = np.vstack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]]).T
    params_dict["quats"] = torch.nn.Parameter(torch.tensor(rotations, dtype=torch.float32, device=device))

    return torch.nn.ParameterDict(params_dict)