# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import os
import logging
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

parser = ArgumentParser("COLMAP reconstruction pipeline")
parser.add_argument("--no_gpu", action="store_true", help="Disable GPU acceleration")
parser.add_argument("--skip_features", action="store_true", help="Skip feature extraction")
parser.add_argument("--skip_matching", action="store_true", help="Skip feature matching")
parser.add_argument("--skip_ba", action="store_true", help="Skip bundle adjustment")
parser.add_argument("--split_ba", action="store_true", help="Use hierarchical mapper instead of regular mapper")
parser.add_argument(
    "--refine_ba", action="store_true", help="Run additional triangulation and BA refinement iterations"
)
parser.add_argument("--refine_iters", default=1, type=int, help="Number of refinement iterations")
parser.add_argument("--global_ba", action="store_true", help="Run global bundle adjustment after mapper")
parser.add_argument("--source_path", "-s", required=True, type=str, help="Path to source data directory")
parser.add_argument("--camera", default="PINHOLE", type=str, help="Camera model (e.g., PINHOLE, OPENCV, RADIAL)")
parser.add_argument("--colmap_executable", default="", type=str, help="Path to COLMAP executable (default: colmap)")
parser.add_argument("--geoalign", action="store_true", help="Align model using GPS/geographic data")
parser.add_argument("--align", action="store_true", help="Align model to plane")
parser.add_argument("--multicamera", action="store_true", help="Use single camera per folder instead of single camera")
parser.add_argument("--max_image_size", default=7096, type=int, help="Maximum image size for feature extraction")
parser.add_argument("--max_sift_features", default=20000, type=int, help="Maximum number of SIFT features per image")
parser.add_argument("--sift_threads", default=14, type=int, help="Number of threads for SIFT extraction")
parser.add_argument("--mask_path", default=None, type=str, help="Path to image masks directory")
parser.add_argument("--random_seed", default=None, type=int, help="Random seed for deterministic reconstruction")
parser.add_argument(
    "--GlobalPositioning_use_gpu",
    type=int,
    default=1,
    help="Set GlobalPositioning.use_gpu flag (0 or 1)",
)
parser.add_argument(
    "--BundleAdjustment_use_gpu",
    type=int,
    default=1,
    help="Set BundleAdjustment.use_gpu flag (0 or 1)",
)

args = parser.parse_args()
colmap_command = args.colmap_executable if args.colmap_executable else "colmap"
use_gpu = 1 if not args.no_gpu else 0

single_camera_tag = "single_camera"
if args.multicamera:
    single_camera_tag = "single_camera_per_folder"

max_image_size = args.max_image_size
sift_max_features = args.max_sift_features
sift_matches = sift_max_features * 3

match_gpu = use_gpu
if sift_max_features > 32768:
    logging.warning("SIFT features > 32768, disabling GPU for matching")
    match_gpu = 0

if not args.skip_features:
    logging.info("Starting feature extraction...")
    fn = os.path.join(args.source_path, "database.db*")
    os.system(f"rm -f {fn}")

    fn = os.path.join(args.source_path, "sparse")
    os.system(f"rm -rf {fn}")

    feat_extracton_cmd = (
        f"{colmap_command} feature_extractor "
        f"--database_path {args.source_path}/database.db "
        f"--image_path {args.source_path}/images_raw "
        f"--ImageReader.camera_model {args.camera} "
        f"--ImageReader.{single_camera_tag} 1 "
        f"--SiftExtraction.use_gpu {use_gpu} "
        f"--SiftExtraction.max_image_size {max_image_size} "
        f"--SiftExtraction.max_num_features {sift_max_features} "
        f"--SiftExtraction.num_threads {args.sift_threads}"
    )

    if args.mask_path is not None:
        feat_extracton_cmd += f" --ImageReader.mask_path {args.mask_path}"

    logging.info(f"Running: {feat_extracton_cmd}")
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)
    logging.info("Feature extraction completed successfully.")

if not args.skip_matching:
    logging.info("Starting feature matching...")
    matcher_type = "exhaustive_matcher"
    feat_matching_cmd = (
        f"{colmap_command} {matcher_type} "
        f"--database_path {args.source_path}/database.db "
        f"--SiftMatching.use_gpu {match_gpu} "
        f"--SiftMatching.max_num_matches {sift_matches} "
        f"--SiftMatching.guided_matching=true"
    )

    logging.info(f"Running: {feat_matching_cmd}")
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)
    logging.info("Feature matching completed successfully.")

sparse_fn = os.path.join(args.source_path, "sparse")
ba_fn = os.path.join(sparse_fn, "0")

if not args.skip_ba:
    logging.info("Starting mapper and bundle adjustment...")
    os.system(f"rm -rf {sparse_fn}")
    os.makedirs(ba_fn)

    if not args.split_ba:
        mapper_cmd = (
            f"{colmap_command} mapper "
            f"--database_path {args.source_path}/database.db "
            f"--image_path {args.source_path}/images_raw "
            f"--output_path {sparse_fn} "
            f"--Mapper.ba_global_use_pba 0 "
            f"--Mapper.ba_global_max_num_iterations 50 "
            f"--BundleAdjustment.use_gpu {args.BundleAdjustment_use_gpu}"
        )

        if args.random_seed is not None:
            mapper_cmd += f" --Mapper.init_min_tri_angle {args.random_seed}"

        logging.info(f"Running: {mapper_cmd}")
        exit_code = os.system(mapper_cmd)
        if exit_code != 0:
            logging.error(f"Mapper failed with code {exit_code}. Exiting.")
            exit(exit_code)
        logging.info("Mapper completed successfully.")
    else:
        mapper_cmd = (
            f"{colmap_command} hierarchical_mapper "
            f"--database_path {args.source_path}/database.db "
            f"--image_path {args.source_path}/images_raw "
            f"--output_path {sparse_fn} "
            f"--Mapper.tri_min_angle 0.75 "
            f"--BundleAdjustment.use_gpu {args.BundleAdjustment_use_gpu}"
        )

        if args.random_seed is not None:
            mapper_cmd += f" --Mapper.init_min_tri_angle {args.random_seed}"

        logging.info(f"Running: {mapper_cmd}")
        exit_code = os.system(mapper_cmd)
        if exit_code != 0:
            logging.error(f"Mapper failed with code {exit_code}. Exiting.")
            exit(exit_code)
        logging.info("Hierarchical mapper completed successfully.")

if args.global_ba and not args.skip_ba:
    logging.info("Running global bundle adjustment...")
    global_ba_cmd = (
        f"{colmap_command} bundle_adjuster "
        f"--input_path {ba_fn} "
        f"--output_path {ba_fn} "
        f"--BundleAdjustment.use_gpu {args.BundleAdjustment_use_gpu}"
    )

    logging.info(f"Running: {global_ba_cmd}")
    exit_code = os.system(global_ba_cmd)
    if exit_code != 0:
        logging.error(f"Global bundle adjustment failed with code {exit_code}. Exiting.")
        exit(exit_code)
    logging.info("Global bundle adjustment completed successfully.")


if args.refine_ba:
    logging.info(f"Running {args.refine_iters} refinement iteration(s)...")
    for i in range(args.refine_iters):
        logging.info(f"Refinement iteration {i+1}/{args.refine_iters}...")

        tri_cmd = (
            f"{colmap_command} point_triangulator "
            f"--input_path {ba_fn} "
            f"--database_path {args.source_path}/database.db "
            f"--image_path {args.source_path}/images_raw "
            f"--output_path {ba_fn} "
            f"--Mapper.tri_min_angle 0.75"
        )

        logging.info(f"Running: {tri_cmd}")
        exit_code = os.system(tri_cmd)
        if exit_code != 0:
            logging.error(f"Triangulator failed with code {exit_code}. Exiting.")
            exit(exit_code)

        bundle_cmd = (
            f"{colmap_command} bundle_adjuster "
            f"--input_path {ba_fn} "
            f"--output_path {ba_fn} "
            f"--BundleAdjustment.use_gpu {args.BundleAdjustment_use_gpu}"
        )

        logging.info(f"Running: {bundle_cmd}")
        exit_code = os.system(bundle_cmd)
        if exit_code != 0:
            logging.error(f"Bundle Adjuster failed with code {exit_code}. Exiting.")
            exit(exit_code)

    logging.info("Refinement completed successfully.")


if args.geoalign or args.align:
    logging.info("Starting model alignment...")
    align_fn = os.path.join(args.source_path, "sparse", "aligned")
    os.system(f"rm -rf {align_fn}")
    os.makedirs(align_fn)

    if args.geoalign:
        ref = "1"
        align_type = "ECEF"
        logging.info("Using geographic alignment (ECEF)")
    else:
        ref = "0"
        align_type = "plane"
        logging.info("Using plane alignment")

    alignment_cmd = (
        f"{colmap_command} model_aligner "
        f"--input_path {ba_fn} "
        f"--output_path {align_fn} "
        f"--database_path {args.source_path}/database.db "
        f"--ref_is_gps {ref} "
        f"--alignment_type {align_type} "
        f"--alignment_max_error 3.0"
    )

    logging.info(f"Running: {alignment_cmd}")
    exit_code = os.system(alignment_cmd)
    if exit_code != 0:
        logging.error(f"Model aligner failed with code {exit_code}. Exiting.")
        exit(exit_code)

    os.system(f"rm -f {ba_fn}/*.bin")
    os.system(f"cp {align_fn}/* {ba_fn}/")
    os.system(f"rm -rf {align_fn}")
    logging.info("Model alignment completed successfully.")

logging.info("Starting image undistortion...")
undistort_cmd = (
    f"{colmap_command} image_undistorter "
    f"--image_path {args.source_path}/images_raw "
    f"--input_path {ba_fn} "
    f"--output_path {args.source_path} "
    f"--output_type=COLMAP"
)

logging.info(f"Running: {undistort_cmd}")
exit_code = os.system(undistort_cmd)
if exit_code != 0:
    logging.error(f"Undistorter failed with code {exit_code}. Exiting.")
    exit(exit_code)

dist_fn = os.path.join(args.source_path, "sparse", "distorted")
os.system(f"cp -r {ba_fn} {dist_fn}")
os.system(f"mv {sparse_fn}/*.bin {ba_fn}/")
logging.info("Image undistortion completed successfully.")
logging.info("COLMAP reconstruction pipeline completed!")
