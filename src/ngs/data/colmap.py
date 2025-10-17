"""COLMAP data parser."""

import json
import os
from pathlib import Path
from typing import Any, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from typing_extensions import assert_never

from ngs.data.read_write_model import read_model

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> list[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, _dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        optimize_foreground: bool = False,
        foreground_margin: float = 0.1,
        load_images_in_memory: bool = False,
        exclude_prefixes: Optional[List[str]] = None,
    ):
        """Initialize the parser.

        :param data_dir: The directory containing the COLMAP data.
        :param factor: The factor to scale the images by.
        :param normalize: Whether to normalize the dataset.
        :param test_every: The frequency of test images.
        :param optimize_foreground: Whether to optimize by cropping to foreground bounding box.
        :param foreground_margin: Margin to add around foreground bounding box (as fraction of box size).
        :param load_images_in_memory: Whether to load all images into memory during initialization.
        :param exclude_prefixes: List of prefixes to exclude images.
        """
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.optimize_foreground = optimize_foreground
        self.foreground_margin = foreground_margin
        self.load_images_in_memory = load_images_in_memory
        self.exclude_prefixes = exclude_prefixes or []

        colmap_dir = os.path.join(data_dir, "sparse/pose_opt_colmap/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        if not os.path.exists(colmap_dir):
            raise ValueError(f"COLMAP directory {colmap_dir} does not exist.")

        # Load COLMAP model
        cameras, images, points3D = read_model(Path(colmap_dir))

        # Extract extrinsic matrices in world-to-camera format.
        w2c_mats = []
        camera_ids = []
        Ks_dict = {}
        params_dict = {}
        imsize_dict = {}  # width, height
        mask_dict = {}
        undist_mask_dict = {}
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

        # Create name to image id mapping
        name_to_image_id = {img.name: img_id for img_id, img in images.items()}

        # If optimizing foreground, we'll create a unique camera for each image
        new_camera_id = 0

        for _image_id, image in images.items():
            rot = image.qvec2rotmat()
            trans = image.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # Get camera parameters from original camera
            original_camera_id = image.camera_id
            cam = cameras[original_camera_id]

            # Use new camera ID if optimizing foreground
            camera_id = new_camera_id if self.optimize_foreground else original_camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            if cam.model == "SIMPLE_PINHOLE":
                fx = fy = cam.params[0]
                cx, cy = cam.params[1:]
            elif cam.model == "PINHOLE":
                fx, fy = cam.params[0:2]
                cx, cy = cam.params[2:]
            else:
                fx = fy = cam.params[0]
                cx, cy = cam.params[1:3]

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters
            if cam.model == "SIMPLE_PINHOLE" or cam.model == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif cam.model == "SIMPLE_RADIAL":
                params = np.array([*cam.params[3:4], 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif cam.model == "RADIAL":
                params = np.array([*cam.params[3:5], 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif cam.model == "OPENCV":
                params = np.array(cam.params[4:8], dtype=np.float32)
                camtype = "perspective"
            elif cam.model == "OPENCV_FISHEYE":
                params = np.array(cam.params[4:8], dtype=np.float32)
                camtype = "fisheye"
            else:
                raise ValueError(f"Unsupported camera model: {cam.model}")

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            undist_mask_dict[camera_id] = None

            if self.optimize_foreground:
                new_camera_id += 1

        print(f"[Parser] {len(images)} images, taken by {len(set(camera_ids))} cameras.")

        w2c_mats = np.stack(w2c_mats, axis=0)
        camtoworlds = np.linalg.inv(w2c_mats)

        # Sort images by name to maintain compatibility with previous behavior
        image_names = [images[k].name for k in images]
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        if self.exclude_prefixes:
            print(f"[Parser] Excluding images with prefixes: {self.exclude_prefixes}")
            keep_indices = []
            original_image_names = image_names
            image_names = []
            for i, name in enumerate(original_image_names):
                if not any(name.startswith(p) for p in self.exclude_prefixes):
                    keep_indices.append(i)
                    image_names.append(name)

            camtoworlds = camtoworlds[keep_indices]
            camera_ids = [camera_ids[i] for i in keep_indices]
            print(f"[Parser] {len(image_names)} images remaining after exclusion.")

            # Filter images and points3D
            kept_image_names = set(image_names)
            kept_image_ids = set()
            new_images = {}
            for img_id, img in images.items():
                if img.name in kept_image_names:
                    new_images[img_id] = img
                    kept_image_ids.add(img_id)
            images = new_images

            new_points3D = {}
            for p_id, p in points3D.items():
                new_image_ids = []
                new_point2D_idxs = []
                for i, img_id in enumerate(p.image_ids):
                    if img_id in kept_image_ids:
                        new_image_ids.append(img_id)
                        new_point2D_idxs.append(p.point2D_idxs[i])
                if len(new_image_ids) > 0:
                    new_points3D[p_id] = p._replace(
                        image_ids=np.array(new_image_ids),
                        point2D_idxs=np.array(new_point2D_idxs),
                    )
            points3D = new_points3D
            print(f"[Parser] {len(points3D)} 3D points remaining after exclusion.")

        # Convert points3D to required format
        points_array = []
        points_colors = []
        points_errors = []
        point3D_id_to_point3D_idx = {}
        point3D_id_to_images = {}

        for idx, (point3D_id, point) in enumerate(points3D.items()):
            points_array.append(point.xyz)
            points_colors.append(point.rgb)
            points_errors.append(point.error)
            point3D_id_to_point3D_idx[point3D_id] = idx
            point3D_id_to_images[point3D_id] = np.column_stack((point.image_ids, point.point2D_idxs))

        points_array = np.array(points_array)
        points_colors = np.array(points_colors)
        points_errors = np.array(points_errors)

        # Load extended metadata
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Handle image loading and path mapping
        # if factor > 1 and not self.extconf["no_factor_suffix"]:
        #     image_dir_suffix = f"_{factor}"
        # else:
        #     image_dir_suffix = ""
        image_dir_suffix = ""

        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files, strict=False))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # Initialize dictionary to store images in memory
        self.images_dict = {}

        # Normalize world space if requested
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points_array = transform_points(T1, points_array)

            T2 = align_principle_axes(points_array)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points_array = transform_points(T2, points_array)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        # Store all attributes
        self.image_names = image_names
        self.image_paths = image_paths
        self.camtoworlds = camtoworlds
        self.camera_ids = camera_ids
        self.Ks_dict = Ks_dict
        self.params_dict = params_dict
        self.imsize_dict = imsize_dict
        self.mask_dict = mask_dict
        self.undist_mask_dict = undist_mask_dict
        self.points = points_array
        self.points_err = points_errors
        self.points_rgb = points_colors
        self.point3D_id_to_point3D_idx = point3D_id_to_point3D_idx
        self.point3D_id_to_images = point3D_id_to_images
        self.name_to_image_id = name_to_image_id
        self.transform = transform

        # Initialize undistortion maps (separate from segmentation masks)
        self.mapx_dict = {}
        self.mapy_dict = {}
        self.roi_undist_dict = {}

        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            if camera_id not in self.Ks_dict:
                raise ValueError(f"Missing K for camera {camera_id}")
            if camera_id not in self.params_dict:
                raise ValueError(f"Missing params for camera {camera_id}")
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(K, params, (width, height), 0)
                mapx, mapy = cv2.initUndistortRectifyMap(K, params, None, K_undist, (width, height), cv2.CV_32FC1)
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = 1.0 + params[0] * theta**2 + params[1] * theta**4 + params[2] * theta**6 + params[3] * theta**8
                mapx = fx * x1 * r + width // 2
                mapy = fy * y1 * r + height // 2

                # Use mask to define ROI for undistortion
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.undist_mask_dict[camera_id] = mask

        # Calculate scene scale from camera positions
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

        #######################################
        # Read masks and save in self.masks_dict
        # Add mask directory path
        # if factor > 1 and not self.extconf["no_factor_suffix"]:
        #     mask_dir_suffix = f"_{factor}"
        # else:
        #     mask_dir_suffix = ""
        mask_dir_suffix = ""
        self.mask_dir = os.path.join(data_dir, "masks" + mask_dir_suffix)

        # Load masks if they exist
        if os.path.exists(self.mask_dir):
            print(f"[Parser] Loading segmentation masks from {self.mask_dir}")
            self.foreground_bboxes = {}  # Store bounding boxes for each image
            for i, image_name in enumerate(image_names):
                mask_path = os.path.join(self.mask_dir, colmap_to_image[image_name])
                if os.path.exists(mask_path):
                    mask = imageio.imread(mask_path)
                    if factor > 1:
                        mask = cv2.resize(
                            mask, (mask.shape[1] // factor, mask.shape[0] // factor), interpolation=cv2.INTER_NEAREST
                        )

                    if len(mask.shape) > 2:  # If mask is RGB/RGBA, convert to binary
                        mask = mask[..., 0]

                    # Get original image dimensions
                    img_path = os.path.join(image_dir, colmap_to_image[image_name])
                    if os.path.exists(img_path):
                        img_shape = imageio.imread(img_path).shape
                        original_height, original_width = img_shape[:2]

                        # # Resize mask to match original image dimensions
                        # if mask.shape[0] != original_height or mask.shape[1] != original_width:
                        #     print(
                        #         f"[Parser] Resizing mask for {image_name} from \
                        #         {mask.shape} to {(original_width, original_height)}"
                        #     )
                        #     mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

                    # Convert to boolean mask
                    mask = mask.astype(np.float32) / 255.0
                    # mask = mask > 0.9  # Convert to boolean

                    # Calculate bounding box if optimizing foreground
                    if self.optimize_foreground:
                        y_indices, x_indices = np.nonzero(mask > 0.1)
                        if len(y_indices) > 0 and len(x_indices) > 0:
                            y_min, y_max = y_indices.min(), y_indices.max() + 1
                            x_min, x_max = x_indices.min(), x_indices.max() + 1

                            # Add margin
                            height = y_max - y_min
                            width = x_max - x_min
                            margin_y = int(height * self.foreground_margin)
                            margin_x = int(width * self.foreground_margin)

                            y_min = max(0, y_min - margin_y)
                            y_max = min(original_height, y_max + margin_y)
                            x_min = max(0, x_min - margin_x)
                            x_max = min(original_width, x_max + margin_x)

                            # Store bounding box
                            self.foreground_bboxes[image_name] = (x_min, y_min, x_max - x_min, y_max - y_min)

                            # Update camera intrinsics for this image
                            camera_id = self.camera_ids[i]
                            K = self.Ks_dict[camera_id].copy()
                            K[0, 2] -= x_min  # Adjust principal point x
                            K[1, 2] -= y_min  # Adjust principal point y
                            self.Ks_dict[camera_id] = K

                            # Update image size
                            self.imsize_dict[camera_id] = (x_max - x_min, y_max - y_min)

                            # Crop the mask to the bounding box
                            mask = mask[y_min:y_max, x_min:x_max]
                        else:
                            print(f"Warning: No foreground found in mask for {image_name}")
                            self.foreground_bboxes[image_name] = (0, 0, original_width, original_height)

                    # Store cropped segmentation mask
                    self.mask_dict[image_name] = mask
        else:
            self.mask_dir = None
            self.mask_dict = {}
            if self.optimize_foreground:
                raise ValueError("optimize_foreground is True but no mask directory exists")

        # Load all images in memory if requested
        if self.load_images_in_memory:
            print(f"[Parser] Loading {len(self.image_paths)} images into memory...")
            for i, (image_name, image_path) in enumerate(zip(self.image_names, self.image_paths, strict=False)):
                camera_id = self.camera_ids[i]
                params = self.params_dict[camera_id]

                # Load and process masks if available
                mask = None
                if self.mask_dir is not None and image_name in self.mask_dict:
                    mask = self.mask_dict[image_name].copy()

                    # Undistort mask if needed
                    if len(params) > 0:
                        mapx, mapy = self.mapx_dict[camera_id], self.mapy_dict[camera_id]
                        mask = cv2.remap(mask, mapx, mapy, cv2.INTER_NEAREST)

                        # Apply ROI cropping to mask
                        x, y, w, h = self.roi_undist_dict[camera_id]
                        mask = mask[y : y + h, x : x + w]

                    # Store fully processed mask
                    self.mask_dict[image_name] = mask

                # Load image and convert to RGB
                image = imageio.imread(image_path)[..., :3]
                image = image.astype(np.float32) / 255.0

                if factor > 1:
                    image = cv2.resize(
                        image, (image.shape[1] // factor, image.shape[0] // factor), interpolation=cv2.INTER_LINEAR
                    )

                if len(params) > 0:
                    # Undistort image if necessary
                    mapx, mapy = self.mapx_dict[camera_id], self.mapy_dict[camera_id]
                    image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

                    # Apply ROI cropping to image
                    x, y, w, h = self.roi_undist_dict[camera_id]
                    image = image[y : y + h, x : x + w]

                # Apply foreground cropping if enabled
                if self.optimize_foreground and self.mask_dir is not None:
                    x, y, w, h = self.foreground_bboxes[image_name]
                    image = image[y : y + h, x : x + w]

                # Apply segmentation mask to image if available
                if mask is not None:
                    image *= np.expand_dims(mask, axis=-1)
                    # # Binarize mask using threshold before applying to image
                    # binary_mask = (mask > 0.0).astype(np.float32)
                    # image *= np.expand_dims(binary_mask, axis=-1)

                # Store processed image in memory
                self.images_dict[image_name] = image
            print("[Parser] Finished loading images and masks into memory.")


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: int | None = None,
        load_depths: bool = False,
    ):
        """Initialize the dataset.

        :param parser: The parser object.
        :param split: The split of the dataset.
        :param patch_size: The size of the patch to crop from the images.
        :param load_depths: Whether to load depths.
        """
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))
        if self.parser.test_every == 0:
            if split == "train":
                self.indices = indices
            else:
                self.indices = []
        else:
            if split == "train":
                self.indices = indices[indices % self.parser.test_every != 0]
            else:
                self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        """Get an item from the dataset.

        :param item: The index of the item in the dataset.
        :return: A dictionary containing the item.
        """
        index = self.indices[item]
        image_name = self.parser.image_names[index]

        # Use preloaded image if available, otherwise load it on demand
        if self.parser.load_images_in_memory and image_name in self.parser.images_dict:
            image = self.parser.images_dict[image_name].copy()
            # Get mask if available - already fully processed during initialization
            mask = self.parser.mask_dict.get(image_name, None) if self.parser.mask_dir is not None else None
        else:
            image = imageio.imread(self.parser.image_paths[index])[..., :3]
            image = image.astype(np.float32) / 255.0
            if self.parser.factor > 1:
                image = cv2.resize(
                    image,
                    (image.shape[1] // self.parser.factor, image.shape[0] // self.parser.factor),
                    interpolation=cv2.INTER_LINEAR,
                )
            camera_id = self.parser.camera_ids[index]
            params = self.parser.params_dict[camera_id]

            # Get mask if available
            if self.parser.mask_dir is not None:
                mask = self.parser.mask_dict[image_name]
            else:
                mask = None

            if len(params) > 0:
                # Images are distorted. Undistort them.
                mapx, mapy = (
                    self.parser.mapx_dict[camera_id],
                    self.parser.mapy_dict[camera_id],
                )
                image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

                # Apply same undistortion to segmentation mask if available
                if mask is not None:
                    mask = cv2.remap(mask, mapx, mapy, cv2.INTER_NEAREST)

                # Apply ROI cropping to both image and segmentation mask
                x, y, w, h = self.parser.roi_undist_dict[camera_id]
                image = image[y : y + h, x : x + w]

                if mask is not None:
                    mask = mask[y : y + h, x : x + w]

            # Apply foreground cropping if enabled
            if self.parser.optimize_foreground and self.parser.mask_dir is not None:
                x, y, w, h = self.parser.foreground_bboxes[image_name]
                image = image[y : y + h, x : x + w]
                # No need to crop mask here as it's already cropped during initialization

            # Apply segmentation mask to image if available
            if mask is not None:
                image *= np.expand_dims(mask, axis=-1)
                # # Binarize mask using threshold before applying to image
                # binary_mask = (mask > 0.1).astype(np.float32)
                # image *= np.expand_dims(binary_mask, axis=-1)

        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        camtoworlds = self.parser.camtoworlds[index]

        # Get undistortion mask if needed
        undist_mask = self.parser.undist_mask_dict[camera_id]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]

            # Also crop the segmentation mask if available
            if mask is not None:
                mask = mask[y : y + self.patch_size, x : x + self.patch_size]

            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).float()

        if undist_mask is not None:
            data["undist_mask"] = torch.from_numpy(undist_mask).bool()

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--load_images_in_memory", action="store_true", help="Load all images into memory")
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8, load_images_in_memory=args.load_images_in_memory
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm.tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
