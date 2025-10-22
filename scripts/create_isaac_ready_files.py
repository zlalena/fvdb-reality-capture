#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import sys
import pathlib
from typing import Dict, Union, Optional
import numpy as np
import point_cloud_utils as pcu
from pathlib import Path
from fvdb import GaussianSplat3d
import torch
import logging
import argparse
from fvdb_reality_capture.tools._export_splats_to_usdz import export_splats_to_usdz


def create_rotation_matrix_x(degrees: float) -> np.ndarray:
    """Create rotation matrix for rotation around X axis."""
    rad = np.radians(degrees)
    cos, sin = np.cos(rad), np.sin(rad)
    return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])


def create_rotation_matrix_z(degrees: float) -> np.ndarray:
    """Create rotation matrix for rotation around Z axis."""
    rad = np.radians(degrees)
    cos, sin = np.cos(rad), np.sin(rad)
    return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])


def apply_isaac_sim_mesh_transform(vertices: np.ndarray) -> np.ndarray:
    """
    Apply Isaac Sim alignment transform to mesh vertices.
    This applies: X rotation 90°, then Z rotation 180°

    Args:
        vertices: Nx3 array of vertex positions

    Returns:
        Transformed Nx3 array of vertex positions
    """
    # First rotate 90° around X axis
    rot_x = create_rotation_matrix_x(90)
    vertices = vertices @ rot_x.T

    # Then rotate 180° around Z axis
    rot_z = create_rotation_matrix_z(180)
    vertices = vertices @ rot_z.T

    return vertices


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions (w, x, y, z format).

    Args:
        q1: First quaternion(s) of shape (..., 4)
        q2: Second quaternion(s) of shape (..., 4)

    Returns:
        Product quaternion(s) of shape (..., 4)
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert a 3x3 rotation matrix to a quaternion (w, x, y, z).

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion as (w, x, y, z)
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return torch.tensor([w, x, y, z], dtype=R.dtype, device=R.device)


def apply_isaac_sim_splat_transform(model: GaussianSplat3d) -> GaussianSplat3d:
    """
    Apply Isaac Sim alignment transform to a Gaussian splat model.
    This applies: X rotation -90°, then X rotation 180° (total -270° or +90° then flip)

    Args:
        model: GaussianSplat3d model to transform

    Returns:
        Transformed GaussianSplat3d model (modified in-place)
    """
    # First: Create rotation matrix for -90° around X axis
    rad_x1 = np.radians(-90)
    cos_x1, sin_x1 = np.cos(rad_x1), np.sin(rad_x1)

    rot_x1 = torch.tensor(
        [[1, 0, 0], [0, cos_x1, -sin_x1], [0, sin_x1, cos_x1]], dtype=model.means.dtype, device=model.device
    )

    # Second: Create rotation matrix for 180° around X axis to flip upside down
    rad_x2 = np.radians(180)
    cos_x2, sin_x2 = np.cos(rad_x2), np.sin(rad_x2)

    rot_x2 = torch.tensor(
        [[1, 0, 0], [0, cos_x2, -sin_x2], [0, sin_x2, cos_x2]], dtype=model.means.dtype, device=model.device
    )

    # Transform positions: first -90° X rotation, then 180° X rotation
    model.means = model.means @ rot_x1.T
    model.means = model.means @ rot_x2.T

    # Transform quaternions
    # Convert rotation matrices to quaternions
    rot_quat_x1 = rotation_matrix_to_quaternion(rot_x1)
    rot_quat_x2 = rotation_matrix_to_quaternion(rot_x2)

    # Apply rotations to all quaternions: q_new = rot_x2 * rot_x1 * q_old
    model.quats = quaternion_multiply(rot_quat_x1.unsqueeze(0), model.quats)
    model.quats = quaternion_multiply(rot_quat_x2.unsqueeze(0), model.quats)

    return model


def crop_and_convert_mesh_to_obj(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    bbox: list[float] | None = None,
    resolution: int = 100_000,
    logger: logging.Logger = logging.getLogger(__name__),
):
    """
    Convert a mesh to watertight format, optionally cropping to a bounding box.

    Args:
        input_path (pathlib.Path): Path to input mesh file (PLY format)
        output_path (pathlib.Path): Path to save the processed mesh
        bbox (list[float], optional): Bounding box coordinates [min_x, min_y, min_z, max_x, max_y, max_z]
    """
    # Load the mesh with vertices and faces
    v, f = pcu.load_mesh_vf(str(input_path))

    logger = logging.getLogger(__name__)
    logger.info("Converting mesh to OBJ")
    if bbox is not None:
        # Unpack bounding box coordinates
        min_x, min_y, min_z, max_x, max_y, max_z = bbox

        # Create mask for vertices within bbox
        mask = (
            (v[:, 0] >= min_x)
            & (v[:, 0] <= max_x)
            & (v[:, 1] >= min_y)
            & (v[:, 1] <= max_y)
            & (v[:, 2] >= min_z)
            & (v[:, 2] <= max_z)
        )

        # Get indices of vertices to keep
        keep_indices = np.where(mask)[0]

        # Create mapping from old vertex indices to new ones
        old_to_new = np.full(v.shape[0], -1)
        old_to_new[keep_indices] = np.arange(len(keep_indices))

        # Filter vertices
        v = v[keep_indices]

        # Filter faces - keep only faces where all vertices are within bounds
        valid_faces = []
        for face in f:
            if all(old_to_new[idx] != -1 for idx in face):
                # Remap vertex indices
                new_face = [old_to_new[idx] for idx in face]
                valid_faces.append(new_face)

        f = np.array(valid_faces, dtype=np.int32)
        # print the new bounds
        print(f"Cropped to:")
        print(f"  min: {v.min(axis=0)}")
        print(f"  max: {v.max(axis=0)}")

    # Make mesh watertight
    # See https://github.com/hjwdzh/Manifold for details
    resolution = resolution
    v_watertight, f_watertight = pcu.make_mesh_watertight(v, f, resolution=resolution)
    print(f"\nWatertight mesh has {v_watertight.shape[0]} vertices and {f_watertight.shape[0]} faces")

    # Convert to the expected types
    v_clean = v_watertight.astype(np.float32)
    f_clean = f_watertight.astype(np.int32)

    # Apply Isaac Sim alignment transform (X: 90°, Z: 180°)
    logger.info("Applying Isaac Sim alignment transform to mesh")
    v_clean = apply_isaac_sim_mesh_transform(v_clean)

    # Write OBJ file
    with open(output_path, "w") as f:
        # Write vertices
        for v in v_clean:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # Write faces (OBJ uses 1-based indexing)
        for face in f_clean:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"Saved watertight mesh to {output_path}")


def crop_and_convert_splat_to_usdz(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    bbox: list[float] | None = None,
    logger: logging.Logger = logging.getLogger(__name__),
):
    """
    Convert a Gaussian splat to USDZ format, optionally cropping to a bounding box.

    Args:
        input_path (pathlib.Path): Path to input splat file (PLY format)
        output_path (pathlib.Path): Path to save the USDZ file
        bbox (list[float], optional): Bounding box coordinates [min_x, min_y, min_z, max_x, max_y, max_z]
        logger (logging.Logger): Logger instance for logging messages
    """
    model, metadata = GaussianSplat3d.from_ply(str(input_path))  # Convert Path to string

    logger.info("Converting Splat to USDZ")

    if bbox is not None:
        # Get positions for cropping (before transformation)
        xyz = model.means.cpu().numpy()
        # Create mask for points within bbox
        min_x, min_y, min_z, max_x, max_y, max_z = bbox

        mask = (
            (xyz[:, 0] >= min_x)
            & (xyz[:, 0] <= max_x)
            & (xyz[:, 1] >= min_y)
            & (xyz[:, 1] <= max_y)
            & (xyz[:, 2] >= min_z)
            & (xyz[:, 2] <= max_z)
        )
        mask = torch.from_numpy(mask).to(model.device)
        # Create new model with only points in bbox using mask indexing
        model = model[mask]
        logger.info(f"Cropped from {len(xyz)} to {len(model.means)} points")

    # Apply Isaac Sim alignment transform (X: -90°, Z: 180°)
    logger.info("Applying Isaac Sim alignment transform to splat")
    model = apply_isaac_sim_splat_transform(model)

    # Create new metadata dictionary with only compatible types
    new_metadata: Optional[Dict[str, Union[str, int, float, torch.Tensor]]] = {
        "sh_degree": int(model.sh_degree),  # Ensure it's an int
    }

    # Add bbox to metadata if provided
    if bbox is not None:
        new_metadata["bbox"] = torch.tensor(bbox, dtype=torch.float32)  # Convert to tensor

    # If original metadata exists, only copy compatible values
    if metadata is not None:
        for key, value in metadata.items():
            # Only copy compatible values
            if isinstance(value, (str, int, float, torch.Tensor)):
                new_metadata[key] = value

    # Export to USDZ
    export_splats_to_usdz(model, output_path)  # export_to_usdz already handles Path objects


def main():
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="Crop a mesh and/or splat model to a given bounding box")

    # Input/Output arguments
    parser.add_argument("--input-splat", type=Path, help="Input splat file (PLY format)")
    parser.add_argument("--output-path", type=Path, help="Output file path (no extension)")
    parser.add_argument("--input-mesh", type=Path, help="Input mesh file (PLY/OBJ format)")
    # add resolution for mesh
    parser.add_argument("--resolution", type=int, help="Resolution for mesh", required=False, default=100_000)
    # Optional bounding box arguments
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=6,
        metavar=("MIN_X", "MIN_Y", "MIN_Z", "MAX_X", "MAX_Y", "MAX_Z"),
        help="Optional bounding box coordinates to crop the model: min_x min_y min_z max_x max_y max_z",
        required=False,
    )

    args = parser.parse_args()

    # Validate that at least one input is provided
    if not args.input_splat and not args.input_mesh:
        parser.error("At least one of --input-splat or --input-mesh must be provided")

    # Process splat if input is provided
    if not args.output_path:
        parser.error("--output-path is required")

    # Create output paths with extensions
    usdz_output_path = args.output_path.with_suffix(".usdz")
    mesh_output_path = args.output_path.with_suffix(".obj")

    # Process splat if input is provided
    if args.input_splat:
        crop_and_convert_splat_to_usdz(args.input_splat, usdz_output_path, args.bbox, logger)

    # Process mesh if input is provided
    if args.input_mesh:
        crop_and_convert_mesh_to_obj(args.input_mesh, mesh_output_path, args.bbox, args.resolution, logger)


if __name__ == "__main__":
    main()
