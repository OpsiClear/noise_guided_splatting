"""Script to evaluate Gaussian splats from PLY files."""

import json
import os
import time
from collections import defaultdict

import imageio
import numpy as np
import torch
import tqdm
import tyro
from gsplat.rendering import rasterization
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from ngs.config import EvaluatorConfig as Config
from ngs.data.colmap import Dataset, Parser
from ngs.utils.utils import rgb_to_sh, save_ply, load_ply


class SplatsEvaluator:
    """Evaluator for Gaussian splats with support for surface/inside visualization."""

    def __init__(self, cfg: Config):
        """Initialize the SplatsEvaluator."""
        self.cfg = cfg
        self.device = cfg.device
        self.result_dir = self.cfg.result_dir
        self.eps = self.cfg.eps
        # Setup output directories
        os.makedirs(self.result_dir, exist_ok=True)
        self.gt_dir = os.path.join(self.result_dir, "gt")
        os.makedirs(self.gt_dir, exist_ok=True)


        # Load data
        if self.cfg.data_dir is None:
            raise ValueError("cfg.data_dir must be provided.")
        self.parser = Parser(
            data_dir=self.cfg.data_dir,
            factor=self.cfg.data_factor,
            test_every=self.cfg.test_every,
        )

        self.valset = Dataset(self.parser, split="val")

        # Initialize metrics
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if self.cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(self.device)
        elif self.cfg.lpips_net == "vgg":
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {self.cfg.lpips_net}")

        # Initialize key state variables
        self.splats = None
        self.app_module = None
        self.num_noise_gaussians = 0

    def save_gt_images(self):
        """Saves the ground truth images from the validation set."""
        print(f"Saving ground truth images to {self.gt_dir}...")
        valloader = torch.utils.data.DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=1)
        for i, data in tqdm.tqdm(enumerate(valloader), total=len(valloader), desc="Saving GT Images"):
            gt_image_tensor = data["image"]  # This is [B, H, W, C], float32, range [0,1]
            # Assuming batch_size is 1 for valloader as is typical
            gt_image_np = gt_image_tensor.squeeze(0).cpu().numpy()
            gt_image_uint8 = (gt_image_np * 255).astype(np.uint8)
            
            if "mask" in data:
                mask_tensor = data["mask"]
                mask_np = mask_tensor.squeeze(0).cpu().numpy()
                mask_uint8 = (mask_np * 255).astype(np.uint8)[..., np.newaxis]
                gt_image_uint8 = np.concatenate((gt_image_uint8, mask_uint8), axis=2)
            
            gt_image_path = os.path.join(self.gt_dir, f"{i:04d}.png")
            imageio.imwrite(gt_image_path, gt_image_uint8)

    def load_surface_splats(self, path):
        """Load surface Gaussian splats from a PLY file."""
        if path.endswith(".ply"):
            if os.path.exists(path):
                self.splats = load_ply(path, self.cfg.sh_degree, self.device)
                if self.splats and "means" in self.splats:
                    print(f"Loaded {len(self.splats['means'])} surface Gaussian splats from PLY file: {path}")
                else:
                    raise ValueError(f"Failed to load splats or 'means' key missing from PLY: {path}")
                return
            else:
                raise ValueError(f"PLY file does not exist: {path}")
        else:
            raise ValueError(f"Unsupported file format: {path}. Only .ply files are supported for surface splats.")

    def add_inside_splats(self, inside_path, color_inside=None, opacity_inside=None):
        """Add inside gaussians to the existing surface splats using load_ply."""
        if self.splats is None:
            raise ValueError("Surface splats must be loaded first with load_surface_splats()")

        if not os.path.exists(inside_path):
            raise ValueError(f"Inside PLY file does not exist: {inside_path}")

        inside_data = load_ply(inside_path, self.cfg.sh_degree, self.device)
        if not inside_data or "means" not in inside_data:
            raise ValueError(f"Failed to load inside splats or 'means' key missing from PLY: {inside_path}")

        num_inside = len(inside_data["means"])
        print(f"Loaded {num_inside} inside gaussians from: {inside_path}")

        eps = np.finfo(np.float32).eps

        # Override opacity if specified
        if opacity_inside is not None:
            op_val = np.clip(opacity_inside, eps, 1.0 - eps)
            logit_op = torch.logit(torch.tensor(op_val, dtype=torch.float32, device=self.device))
            inside_data["opacities"] = torch.full((num_inside,), logit_op, dtype=torch.float32, device=self.device)
        
        # Determine color to set for inside splats
        color_to_set = color_inside if color_inside is not None else [0.0, 1.0, 0.0] # Default green

        # Override colors based on SH or RGB format (consistent due to load_ply using same sh_degree)
        if "sh0" in inside_data and "sh0" in self.splats:
            inside_sh_coeffs = rgb_to_sh(torch.tensor(color_to_set, dtype=torch.float32, device=self.device))
            
            sh0_values = torch.zeros((num_inside, 1, 3), dtype=torch.float32, device=self.device)
            sh0_values[:, 0, 0] = inside_sh_coeffs[0]
            sh0_values[:, 0, 1] = inside_sh_coeffs[1]
            sh0_values[:, 0, 2] = inside_sh_coeffs[2]
            inside_data["sh0"] = sh0_values

            # Higher order SH coefficients for inside splats should be zero for a solid color
            if "shN" in inside_data:
                k_minus_1 = inside_data["shN"].shape[1]
                inside_data["shN"] = torch.zeros((num_inside, k_minus_1, 3), dtype=torch.float32, device=self.device)
            elif "shN" in self.splats: # Match structure if inside_data didn't have shN initially but surface does
                k_minus_1 = self.splats["shN"].shape[1]
                inside_data["shN"] = torch.zeros((num_inside, k_minus_1, 3), dtype=torch.float32, device=self.device)
            
        elif "colors" in inside_data and "colors" in self.splats:
            rgb_values = torch.zeros((num_inside, 3), dtype=torch.float32, device=self.device)
            rgb_values[:, 0] = color_to_set[0]
            rgb_values[:, 1] = color_to_set[1]
            rgb_values[:, 2] = color_to_set[2]
            inside_data["colors"] = torch.logit(torch.clamp(rgb_values, eps, 1.0 - eps))

            # If features are used, zero them out for inside splats
            if "features" in inside_data and "features" in self.splats:
                feature_dim = inside_data["features"].shape[1]
                inside_data["features"] = torch.zeros((num_inside, feature_dim), dtype=torch.float32, device=self.device)
        else:
            # This case should ideally not happen if load_ply is consistent for surface and inside
            print("Warning: Mismatch in color format (SH/RGB) between surface and inside splats, or format not found.")

        # Concatenate parameters (inside_gaussians first, then surface_gaussians)
        new_params = {}
        all_keys = set(self.splats.keys()) | set(inside_data.keys())
        
        for key in sorted(list(all_keys)):
            if key in inside_data and key in self.splats:
                # Ensure both are tensors. self.splats[key] is a Parameter, so use .data
                new_params[key] = torch.nn.Parameter(
                    torch.cat([inside_data[key], self.splats[key].data], dim=0)
                )
            elif key in self.splats: # Key only in surface (e.g. if inside PLY was minimal)
                # This might indicate an issue or a need to initialize for inside splats
                print(f"Warning: Key '{key}' found in surface splats but not in loaded inside splats. Retaining surface splats' data for this key.")
                new_params[key] = self.splats[key] # Keep existing surface data
            elif key in inside_data:
                 print(f"Warning: Key '{key}' found in loaded inside splats but not in surface splats. This key will be added.")
                 new_params[key] = torch.nn.Parameter(inside_data[key])

        self.splats = torch.nn.ParameterDict(new_params)
        self.num_noise_gaussians = num_inside

    @torch.no_grad()
    def colorize_surface_splats(self, color_surface):
        """Change the color of surface gaussians to a specific color."""
        if self.splats is None:
            raise ValueError("Surface splats must be loaded first with load_surface_splats()")

        # Get the index of the first surface gaussian
        surface_start_idx = self.num_noise_gaussians

        # Make a copy of the parameters for modification
        new_params = {k: torch.nn.Parameter(v.clone()) for k, v in self.splats.items()}

        # Convert color to float tensor
        eps = np.finfo(np.float32).eps
        color_surface = np.array(color_surface)

        if "sh0" in new_params:
            # For SH representation
            sh_color = rgb_to_sh(color_surface)

            # Modify the first SH coefficient (DC term) to the specified color
            new_params["sh0"][surface_start_idx:, 0, 0] = sh_color[0]  # R
            new_params["sh0"][surface_start_idx:, 0, 1] = sh_color[1]  # G
            new_params["sh0"][surface_start_idx:, 0, 2] = sh_color[2]  # B

            # Zero out higher-order SH coefficients for solid color
            new_params["shN"][surface_start_idx:] = 0.0
        else:
            # For RGB color representation
            # Apply logit transformation as we use sigmoid in rasterization
            color_tensor = torch.tensor(color_surface, device=self.device)
            color_tensor = torch.logit(torch.clamp(color_tensor, eps, 1.0 - eps))

            # Set all surface gaussians to the specified color
            new_params["colors"][surface_start_idx:, 0] = color_tensor[0]  # R
            new_params["colors"][surface_start_idx:, 1] = color_tensor[1]  # G
            new_params["colors"][surface_start_idx:, 2] = color_tensor[2]  # B

        # Update the splats
        self.splats = torch.nn.ParameterDict(new_params)
        print(f"Changed color of {len(self.splats['means']) - self.num_noise_gaussians} surface gaussians")

    @torch.no_grad()
    def rasterize_splats(
        self,
        camtoworlds: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Rasterize the splats."""
        means = self.splats["means"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]


        if "sh0" in self.splats:
            # Use SH coefficients
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)  # [N, K, 3]
        else:
            # Simple RGB colors
            colors = torch.sigmoid(self.splats["colors"])  # [N, 3]
            # Reshape for rasterization
            colors = colors.unsqueeze(1)  # [N, 1, 3]


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
            absgrad=False,
            **kwargs,
        )

        return render_colors, render_alphas, info

    @torch.no_grad()
    def evaluate(self, output_prefix="", transparency=False, crop=False):
        """Evaluate the loaded Gaussian splats against the validation set."""
        print("Running evaluation...")
        torch.cuda.empty_cache()

        total_time = 0.0
        metrics = defaultdict(list)

        valloader = torch.utils.data.DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=1)
        output_dir = os.path.join(self.result_dir, output_prefix)
        os.makedirs(output_dir, exist_ok=True)
        if transparency:
            transmittance_dir = os.path.join(self.result_dir, "transmittance")
            os.makedirs(transmittance_dir, exist_ok=True)

        for i, data in tqdm.tqdm(enumerate(valloader), total=len(valloader)):
            # Use the correct key name: "camtoworld" instead of "c2w"
            camtoworlds = data["camtoworld"].to(self.device)
            Ks = data["K"].to(self.device)
            pixels = data["image"].to(self.device)
            masks = data["mask"].to(self.device) if "mask" in data else None

            height, width = pixels.shape[1:3]

            # Time the rendering
            torch.cuda.synchronize()
            tic = time.time()

            # Render the image
            colors, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=self.cfg.sh_degree,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
            )

            torch.cuda.synchronize()
            render_time = time.time() - tic
            total_time += render_time

            # Clamp colors to valid range
            colors = torch.clamp(colors, 0.0, 1.0)

            # Apply masks if available
            if masks is not None:
                # Apply mask
                # colors = colors.clone()
                colors *= masks.unsqueeze(-1)
                # Binarize mask using threshold before applying to image
                binary_mask = (masks > 0.5).float()
                colors *= binary_mask.unsqueeze(-1)
                if crop:
                    # Get indices of True values, handling batch dimension
                    batch_indices, y_indices, x_indices = torch.where(masks > 0.1)

                    # Get min/max coordinates for each batch
                    x_min = x_indices.min().item()
                    x_max = x_indices.max().item()
                    y_min = y_indices.min().item()
                    y_max = y_indices.max().item()

                    colors = colors[:, y_min:y_max, x_min:x_max]
                    pixels = pixels[:, y_min:y_max, x_min:x_max]

                # Crop to mask bounds
                eval_colors = colors
                eval_pixels = pixels

            else:
                eval_colors = colors
                eval_pixels = pixels

            image_np  = colors.squeeze(0).cpu().numpy()
            image_uint8 = (image_np * 255).astype(np.uint8)

            if masks is not None:
                mask_np = masks.squeeze(0).cpu().numpy()
                mask_uint8 = (mask_np * 255).astype(np.uint8)[..., np.newaxis]
                image_uint8 = np.concatenate((image_uint8, mask_uint8), axis=2)

            # Save rendered image
            render_path = f"{output_dir}/{i:04d}.png"
            imageio.imwrite(render_path, image_uint8)

            if transparency:
                # Also save green channel for visualization
                green_channel = colors[..., 1]
                transmittance_map_np = green_channel.squeeze().cpu().numpy()
                transmittance_map = (transmittance_map_np * 255).astype(np.uint8)
                avg_transmittance_value = float(np.sum(transmittance_map_np) / np.sum(masks.cpu().numpy()))
                sos = np.log10(avg_transmittance_value + self.eps) / np.log10(self.eps)
                transmittance_path = f"{transmittance_dir}/{i:04d}.png"
                
                image_to_save = transmittance_map
                if masks is not None:
                    mask_np = masks.squeeze(0).cpu().numpy()
                    mask_uint8 = (mask_np * 255).astype(np.uint8)[..., np.newaxis]
                    transmittance_rgb = np.stack([transmittance_map, transmittance_map, transmittance_map], axis=-1)
                    image_to_save = np.concatenate((transmittance_rgb, mask_uint8), axis=2)
                
                imageio.imwrite(transmittance_path, image_to_save)

            # Calculate metrics
            pixels_p = eval_pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors_p = eval_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            if transparency:
                metrics["sos"].append(sos)
            else:
                metrics["psnr"].append(self.psnr(colors_p, pixels_p).item())
                metrics["ssim"].append(self.ssim(colors_p, pixels_p).item())
                metrics["lpips"].append(self.lpips(colors_p, pixels_p).item())

        # Calculate average metrics
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        avg_metrics["num_gaussians"] = len(self.splats["means"])
        avg_metrics["num_surface_gaussians"] = len(self.splats["means"]) - self.num_noise_gaussians
        avg_metrics["num_inside_gaussians"] = self.num_noise_gaussians

        # Save metrics to file
        stats_file = f"{self.result_dir}/{output_prefix}_metrics.json"
        with open(stats_file, "w") as f:
            json.dump(avg_metrics, f, indent=2)

        return avg_metrics


def main():
    """Main function to evaluate Gaussian splats."""
    cfg = tyro.cli(Config)

    # Ensure required fields from config are present (previously handled by argparse required=True)
    if cfg.data_dir is None:
        print("Error: --data_dir must be specified.")
        return 
    if cfg.surface_pcloud is None:
        print("Error: --surface_pcloud must be specified.") # Tyro converts snake_case to kebab-case for CLI
        return


    # Create evaluator
    evaluator = SplatsEvaluator(cfg)

    # Save ground truth images once before starting evaluations
    evaluator.save_gt_images()

    # Load surface gaussians
    evaluator.load_surface_splats(cfg.surface_pcloud) # Use cfg.surface_pcloud

    # Evaluate surface gaussians alone
    stats = evaluator.evaluate(output_prefix="surface", transparency=False)
    print(stats)

    # If inside gaussians path is provided, add them and evaluate
    if cfg.inside_pcloud is not None: 
        # Add inside gaussians (green)
        evaluator.add_inside_splats(cfg.inside_pcloud, color_inside=[0.0, 1.0, 0.0], opacity_inside=1.0)

        # Evaluate combined model
        stats = evaluator.evaluate(output_prefix="infill", transparency=False)
        print(stats)
        save_ply(evaluator.splats, os.path.join(cfg.result_dir, "infill.ply"))

        # Colorize surface gaussians to red for better visualization
        evaluator.colorize_surface_splats([1.0, 0.0, 0.0])

        # Evaluate with red surface and green inside
        stats = evaluator.evaluate(output_prefix="transparency", transparency=True)
        print(stats)
        save_ply(evaluator.splats, os.path.join(cfg.result_dir, "transparency.ply"))


if __name__ == "__main__":
    main()
