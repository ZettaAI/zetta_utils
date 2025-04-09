from __future__ import annotations

from itertools import chain
from typing import Callable, Literal, Sequence

import attrs
import cv2
import numpy as np

from zetta_utils import builder, log
from zetta_utils.layer import volumetric
from zetta_utils.tensor_ops import common, convert, generators
from zetta_utils.tensor_typing import TensorTypeVar

logger = log.get_logger("zetta_utils")


@builder.register("Transform2D")
@attrs.mutable
class Transform2D:
    """
    Class representing a 2D transformation.

    :param num_octaves: Number of octaves for SIFT feature extraction.
    :param contrast_threshold: Contrast threshold for SIFT feature extraction.
    :param edge_threshold: Edge threshold for SIFT feature extraction.
    :param sigma: Sigma value for SIFT feature extraction.
    :param ratio_test_fraction: Fraction for ratio test in SIFT matching.
    :param num_min_matches: Minimum number of matches required for transformation estimation.
    :param ransac_dist_threshold: RANSAC distance threshold for transformation estimation.
    :param transformation_mode: Mode of transformation. Can be one of "rigid",
                                "partial_affine", "full_affine", or "perspective".
    :param estimate_mode: Mode of estimation. Can be one of "ransac" or "lmeds".
    :returns: 3x3 transformation matrix.
    :raises ValueError: If the estimate mode is not "ransac" or "lmeds".
    :raises ValueError: If the transformation mode is not "rigid", "partial_affine",
                        "full_affine", or "perspective".
    """

    num_octaves: int = 3
    contrast_threshold: float = 0.04
    edge_threshold: float = 10
    sigma: float = 1.6
    ratio_test_fraction: float = 0.7
    num_min_matches: int = 5
    ransac_dist_threshold: float = 3.0
    transformation_mode: Literal["rigid", "partial_affine", "full_affine", "perspective"] = "rigid"
    estimate_mode: Literal["ransac", "lmeds"] = "lmeds"
    ensure_scale_boundaries: tuple[float, float] = (0.91, 1.1)
    allow_flip: bool = False
    _estimate_mode: int = attrs.field(init=False)
    _estimate_fn: Callable = attrs.field(init=False)

    def __attrs_post_init__(self):
        if self.estimate_mode == "ransac":
            self._estimate_mode = cv2.RANSAC
        elif self.estimate_mode == "lmeds":
            self._estimate_mode = cv2.LMEDS
        else:
            raise ValueError(
                f"Expected estimate mode 'ransac' or 'lmeds', got {self.estimate_mode}"
            )

        if self.transformation_mode == "rigid":
            self._estimate_fn = cv2.estimateAffinePartial2D
        elif self.transformation_mode == "partial_affine":
            self._estimate_fn = cv2.estimateAffinePartial2D
        elif self.transformation_mode == "full_affine":
            self._estimate_fn = cv2.estimateAffine2D
        elif self.transformation_mode == "perspective":
            self._estimate_fn = cv2.findHomography
        else:
            raise ValueError(
                f"""Expected transformation mode 'rigid', 'partial_affine', 'full_affine',
                or 'perspective', got {self.transformation_mode}"""
            )

    def __call__(  # pylint: disable=R0911
        self,
        src: TensorTypeVar,  # (C, X, Y, Z)
        tgt: TensorTypeVar,  # (C, X, Y, Z)
    ) -> volumetric.VolumetricLayerDType:
        """
        Apply the transformation to the source tensor.

        :param src: Source tensor.
        :param tgt: Target tensor.
        :return: The estimated 3x3 transformation matrix.
        """
        try:
            device = src.device  # type: ignore
        except AttributeError:
            device = None

        INVALID_TRANSFORM = volumetric.to_vol_layer_dtype(
            np.zeros((1, 3, 3, 1), dtype=np.float32), device=device
        )

        src_np = convert.to_np(src)
        tgt_np = convert.to_np(tgt)
        assert src_np.shape == tgt_np.shape
        assert src_np.dtype == tgt_np.dtype == np.uint8
        src_np = common.rearrange(src_np, "1 X Y 1 -> X Y")
        tgt_np = common.rearrange(tgt_np, "1 X Y 1 -> X Y")

        features = create_sift_keypoints_and_descriptors(
            src=src_np,
            tgt=tgt_np,
            num_octaves=self.num_octaves,
            contrast_threshold=self.contrast_threshold,
            edge_threshold=self.edge_threshold,
            sigma=self.sigma,
        )
        src_kp, tgt_kp, src_desc, tgt_desc = features

        if len(src_kp) == 0:
            logger.info("No keypoints in source image identified, returning identity")
            return INVALID_TRANSFORM

        if len(tgt_kp) == 0:
            logger.info("No keypoints in target image identified, returning identity")
            return INVALID_TRANSFORM

        matches = match_sift_descriptors(src_desc=src_desc, tgt_desc=tgt_desc)

        good_matches = filter_matches_ratio_test(
            matches, ratio_test_fraction=self.ratio_test_fraction
        )

        logger.info(f"Found {len(good_matches)} good matches out of {len(matches)} candidates.")
        if len(good_matches) < self.num_min_matches:
            logger.info("Too few matches to estimate affine transform")
            return INVALID_TRANSFORM

        # Convert keypoint positions to standard coordinates [-1, 1]
        # Also, I think that OpenCV uses center of corner pixels as reference frame,
        # rather than corner of corner pixels, thus the `+ 1`
        max_shape = max(src_np.shape)
        for keypoint in chain(src_kp, tgt_kp):
            keypoint.pt = (
                2.0 * ((keypoint.pt[0] + 1) / (max_shape + 1)) - 1.0,
                2.0 * ((keypoint.pt[1] + 1) / (max_shape + 1)) - 1.0,
            )

        src_pts = np.array(
            [src_kp[m.queryIdx].pt for m in good_matches], dtype=np.float32
        ).reshape(-1, 2)

        tgt_pts = np.array(
            [tgt_kp[m.trainIdx].pt for m in good_matches], dtype=np.float32
        ).reshape(-1, 2)

        mat, mask = self._estimate_fn(
            src_pts,
            tgt_pts,
            method=self._estimate_mode,
            ransacReprojThreshold=self.ransac_dist_threshold,
        )

        if mat is None:
            logger.info("Failed to estimate transformation.")
            return INVALID_TRANSFORM

        # Verify scale factor is within plausible range
        det = np.linalg.det(mat[:2, :2])
        if self.allow_flip:
            det = abs(det)
        logger.info(mat)
        logger.info(det)

        if det < self.ensure_scale_boundaries[0] or det > self.ensure_scale_boundaries[1]:
            logger.info(
                f"Scale factor ({det}) is not within range {self.ensure_scale_boundaries}."
            )
            return INVALID_TRANSFORM

        # Estimate rigid transformation from the partial affine matches
        if self.transformation_mode == "rigid":
            mat = compute_rigid_transform(
                matches=good_matches, mask=mask, src_kp=src_kp, tgt_kp=tgt_kp
            )

        if mat is None:
            logger.info("Failed to estimate rigid transformation.")
            return INVALID_TRANSFORM

        # Ensure 3x3 matrix
        if mat.shape == (2, 3):
            mat = np.vstack([mat, [0, 0, 1]])

        logger.info(mat)
        return convert.to_float32(volumetric.to_vol_layer_dtype(mat).reshape(1, 3, 3, 1))


@builder.register("FieldFromTransform2D")
@attrs.mutable
class FieldFromTransform2D:
    shape: tuple[int, int]

    def __call__(
        self,
        mat: TensorTypeVar,  # (1, 3, 3, 1)
    ) -> volumetric.VolumetricLayerDType:
        assert mat.shape == (1, 3, 3, 1)
        mat_torch = convert.to_torch(mat)

        field = (
            generators.get_field_from_matrix(
                mat.squeeze(-1),
                size=max(self.shape),
                device=mat_torch.device,
            )
            .pixels()[..., : self.shape[0], : self.shape[1]]
            .tensor_()
        )
        return volumetric.to_vol_layer_dtype(common.rearrange(field, "1 C X Y -> C X Y 1"))


def create_sift_keypoints_and_descriptors(
    src: np.ndarray,
    tgt: np.ndarray,
    num_octaves: int = 3,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10,
    sigma: float = 1.0,
) -> tuple[list[cv2.KeyPoint], list[cv2.KeyPoint], np.ndarray, np.ndarray]:
    """Create SIFT features.

    :param src: Source image (HxW).
    :param tgt: Target image (HxW).
    :param num_octaves: Number of octaves. See cv2.SIFT_create documentation.
    :param contrast_threshold: Contrast threshold. See cv2.SIFT_create documentation.
    :param edge_threshold: Edge threshold. See cv2.SIFT_create documentation.
    :param sigma: Sigma. See cv2.SIFT_create documentation.
    :return: Tuple of lists of src & tgt keypoints, then src & tgt descriptors.
    """
    # Initiate SIFT detector
    detector = cv2.SIFT_create(  # type: ignore[attr-defined]
        nOctaveLayers=num_octaves,
        edgeThreshold=edge_threshold,
        contrastThreshold=contrast_threshold,
        sigma=sigma,
        enable_precise_upscale=True,
    )
    # find the keypoints and descriptors with SIFT
    src_kp, src_desc = detector.detectAndCompute(src, None)
    tgt_kp, tgt_desc = detector.detectAndCompute(tgt, None)

    return src_kp, tgt_kp, src_desc, tgt_desc


def match_sift_descriptors(
    src_desc: np.ndarray, tgt_desc: np.ndarray
) -> Sequence[Sequence[cv2.DMatch]]:
    """Match features based on descriptors

    :param src_desc: Source descriptors (num_features x feature_length).
    :param tgt_desc: Target descriptors (num_features x feature_length).
    :return: Matches.
    """
    matcher = cv2.FlannBasedMatcher()
    return matcher.knnMatch(src_desc, tgt_desc, k=2)


def filter_matches_ratio_test(
    matches: Sequence[Sequence[cv2.DMatch]], ratio_test_fraction: float = 0.7
) -> list[cv2.DMatch]:
    """
    Collect good matches per Lowe's ratio test

    :param matches: List of tuples of descriptor pairs.
    :param ratio_test_fraction: [0,1] for Lowe's ratio test.
    :return: Accepted matches.
    """
    good: list[cv2.DMatch] = []
    for desc_a, desc_b in matches:
        if desc_a.distance < ratio_test_fraction * desc_b.distance:
            good.append(desc_a)
    return good


def rigid_transform_3d(src_pts: np.ndarray, tgt_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # pylint: disable=invalid-name

    # From: https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
    # Input: expects 3xN matrix of points
    # Returns R,t
    # R = 3x3 rotation matrix
    # t = 3x1 column vector
    assert len(src_pts) == len(tgt_pts)

    num_rows, num_cols = src_pts.shape

    if num_rows != 3:
        raise ValueError(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    [num_rows, num_cols] = tgt_pts.shape
    if num_rows != 3:
        raise ValueError(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_a = np.mean(src_pts, axis=1)
    centroid_b = np.mean(tgt_pts, axis=1)

    # ensure centroids are 3x1 (necessary when A or B are
    # numpy arrays instead of numpy matrices)
    centroid_a = centroid_a.reshape(-1, 1)
    centroid_b = centroid_b.reshape(-1, 1)

    # subtract mean
    mat_a = src_pts - np.tile(centroid_a, (1, num_cols))
    mat_b = tgt_pts - np.tile(centroid_b, (1, num_cols))

    cross_covar = mat_a @ np.transpose(mat_b)

    # find rotation
    U, _, Vt = np.linalg.svd(cross_covar)
    rotation_mat = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(rotation_mat) < 0:
        Vt[2, :] *= -1
        rotation_mat = Vt.T @ U.T

    translation = (-rotation_mat @ centroid_a) + centroid_b

    return rotation_mat, translation


def compute_rigid_transform(
    matches: list[cv2.DMatch],
    mask: np.ndarray,
    src_kp: list[cv2.KeyPoint],
    tgt_kp: list[cv2.KeyPoint],
) -> np.ndarray:
    """Compute rigid transform (rotation, translation)

    :param matches: List of match objects.
    :param mask: Numpy array for inliers.
    :param src_kp: List of keypoint objects.
    :param tgt_kp: List of keypoint objects.
    :return: Numpy array (3x3) for affine matrix.
    """
    src_pts = np.array(
        [tuple(src_kp[m.queryIdx].pt) + (1,) for i, m in zip(mask, matches) if i], dtype=np.float32
    )
    tgt_pts = np.array(
        [tuple(tgt_kp[m.trainIdx].pt) + (1,) for i, m in zip(mask, matches) if i], dtype=np.float32
    )
    src_pts = src_pts.T
    tgt_pts = tgt_pts.T
    rotation_mat, translation = rigid_transform_3d(src_pts=src_pts, tgt_pts=tgt_pts)  # B = R*A + t
    rotation_mat[:2, 2] += translation[:2, 0]
    assert np.array_equal(rotation_mat[2, :], [0, 0, 1])
    return rotation_mat
