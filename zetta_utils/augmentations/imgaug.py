from __future__ import annotations

import collections
from typing import Any, Final, Literal, Sequence, Sized, Tuple, TypeVar, overload

import numpy as np
import torch
from imgaug import augmenters as iaa
from imgaug.augmenters.meta import Augmenter
from numpy.typing import NDArray

from zetta_utils import builder
from zetta_utils.tensor_ops import common, convert, crop_center
from zetta_utils.tensor_typing import Tensor, TensorTypeVar

SizedTypeVar = TypeVar("SizedTypeVar", bound=Sized)
TensorListTypeVar = TypeVar("TensorListTypeVar", Tensor, Sequence)
T = TypeVar("T")

SUFFIX_MAPPING: Final = {
    "img": "images",
    "seg": "segmentation_maps",
    "mask": "segmentation_maps",
    "hm": "heatmaps",
    "aff": "heatmaps",
    "kp": "keypoints",
    "bb": "bounding_boxes",
    "poly": "polygons",
    "ls": "line_strings",
}

FakeResult = collections.namedtuple(
    "FakeResult", ["images_aug", "segmentation_maps_aug", "heatmaps_aug"]
)


def _ensure_list(augmenter: Augmenter | Sequence[Augmenter]) -> Sequence[Augmenter]:
    return augmenter if isinstance(augmenter, Sequence) else [augmenter]


@overload
def _ensure_nxyc(x: Tensor) -> NDArray:
    ...


@overload
def _ensure_nxyc(x: Sequence[Tensor]) -> list[NDArray]:
    ...


def _ensure_nxyc(x: Tensor | Sequence[Tensor]) -> NDArray | list[NDArray]:
    if isinstance(x, Sequence):
        return [convert.to_np(common.rearrange(v, pattern="C X Y 1 -> X Y C")) for v in x]
    else:
        return convert.to_np(common.rearrange(x, pattern="C X Y N -> N X Y C"))


@overload
def _ensure_cxyn(x: Tensor, ref: TensorTypeVar) -> TensorTypeVar:
    ...


@overload
def _ensure_cxyn(x: Sequence[Tensor], ref: Sequence[TensorTypeVar]) -> list[TensorTypeVar]:
    ...


def _ensure_cxyn(x, ref):
    if isinstance(ref, Sequence):
        return [
            convert.astype(common.rearrange(v, pattern="X Y C -> C X Y 1"), reference=ref[i])
            for i, v in enumerate(x)
        ]
    else:
        return convert.astype(common.rearrange(x, pattern="N X Y C -> C X Y N"), reference=ref)


def _group_kwargs(
    **kwargs: TensorListTypeVar | None,
) -> tuple[dict[str, dict[str, TensorListTypeVar]], dict[str, str]]:
    groups: dict[str, dict[str, TensorListTypeVar]] = {}
    kwarg_mapping: dict[str, str] = {}
    for k, v in kwargs.items():
        if v is None:
            continue
        group_name, suffix = k.rsplit("_", 1)
        try:
            mapped_suffix = SUFFIX_MAPPING[suffix]
            kwarg_mapping[k] = mapped_suffix
        except KeyError as e:
            raise ValueError(
                f"Expected suffix `_img`, `_seg`, or `_aff` in custom augmentable key {k}"
            ) from e

        groups.setdefault(group_name, {})[mapped_suffix] = v

    return groups, kwarg_mapping


def _ungroup_kwargs(
    groups: dict[str, dict[str, TensorListTypeVar]], kwarg_mapping: dict[str, str]
) -> dict[str, TensorListTypeVar]:
    kwargs = {}
    for k, mapped_suffix in kwarg_mapping.items():
        group_name, suffix = k.rsplit("_", 1)
        if group_name == "_args":
            kwargs[mapped_suffix] = groups[group_name][mapped_suffix]
        else:
            kwargs[f"{group_name}_{suffix}"] = groups[group_name][mapped_suffix]

    return kwargs


@builder.register("imgaug_readproc")
def imgaug_readproc(
    *args,  # the zetta_utils builder puts the layer/layerset as the first argument
    targets: Container[str] | None = None,
    **kwargs,  # and the augmenters in the kwargs
):
    assert len(args) == 1
    augmenters = kwargs.pop("augmenters", None)
    assert augmenters is not None
    if isinstance(args[0], dict):
        if targets is not None:
            all_inputs = args[0]
            inputs = {k: all_inputs[k] for k in targets}
            results = imgaug_augment(augmenters, **inputs, **kwargs)
            return {k: results[k] if k in results else all_inputs[k] for k in all_inputs.keys()}
        else:
            return imgaug_augment(augmenters, **args[0], **kwargs)

    else:  # Tensor
        return imgaug_augment(augmenters, images=args[0], **kwargs)["images"]


def imgaug_augment(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    augmenters: Augmenter | Sequence[Augmenter],
    mode: Literal["2d", "3d"] = "2d",
    *,
    images: Tensor | Sequence[Tensor] | None = None,
    heatmaps: Tensor | Sequence[Tensor] | None = None,
    segmentation_maps: Tensor | Sequence[Tensor] | None = None,
    keypoints: Sequence[Sequence[Tuple[float, float]]] | None = None,
    bounding_boxes: Sequence[Sequence[Tuple[float, float, float, float]]] | None = None,
    polygons: Sequence[Sequence[Sequence[Tuple[float, float]]]] | None = None,
    line_strings: Sequence[Sequence[Sequence[Tuple[float, float]]]] | None = None,
    **kwargs: Tensor | Sequence[Tensor],
) -> dict[str, Any]:
    """This function is a wrapper for imgaug.augment to handle the CXYZ/ZXYC conversion.
    It will call each provided augmenter on the provided augmentable dict.

    For additionally supported types, see:
    https://github.com/aleju/imgaug/blob/0101108d4fed06bc5056c4a03e2bcb0216dac326/imgaug/augmenters/meta.py#L1757-L1842

    :param augmenters: A sequence of imgaug augmenters.
    :param images: Either CXYZ tensor or list of CXY1 tensors. If not specified,
      at least one kwarg with `_img` suffix is required.
    :param heatmaps: Either CXYZ tensor or list of CXY1 tensors.
    :param segmentation_maps: Either CXYZ tensor or list of CXY1 tensors.
    :param keypoints: List of lists of (x, y) coordinates.
    :param bounding_boxes: List of lists of (x1, y1, x2, y2) coordinates.
    :param polygons: List of lists of lists of (x, y) coordinates.
    :param line_strings: List of lists of lists of (x, y) coordinates.
    :param kwargs: Additional/alternative augmentables, each a CXYZ tensor or list of CXY1 tensors
      and suffixes: `_img`, `_seg`, `_hm`/`_aff`, `_kp`, `_bb`, `_poly`, `_ls`.

    :return: Augmented dictionary, same keys as input.
    """
    augmenter: Augmenter = iaa.Sequential(_ensure_list(augmenters)).to_deterministic()
    augmentables, kwarg_mapping = _group_kwargs(  # type: ignore
        _args_img=images,
        _args_hm=heatmaps,
        _args_seg=segmentation_maps,
        _args_kp=keypoints,
        _args_bb=bounding_boxes,
        _args_poly=polygons,
        _args_ls=line_strings,
        **kwargs,
    )

    if mode == "3d":
        unsupported_keys = set(["keypoints", "bounding_boxes", "polygons", "line_strings"])
        for group in augmentables.values():
            unsupported_present = unsupported_keys & set(group.keys())
            if len(unsupported_present):
                raise ValueError("3d mode does not support {unsupported_present}")

    # since imgaug always require an image input, we find a ref in case some inputs don't have one
    ref_images = None
    for group in augmentables.values():
        if "images" in group:
            images_ = group["images"]
            assert isinstance(images_, np.ndarray | torch.Tensor | Sequence)
            # mypy seems to have a bug where checking against a union_object doesn't work
            # e.g., the below will not pass checks
            # a = np.ndarray | torch.Tensor | Sequence[np.ndarray | torch.Tensor]
            # assert isinstance(images_, a)
            ref_images = _ensure_nxyc(images_)
            break
    if ref_images is None:
        raise ValueError("Expected at least one image in `images` or `kwargs`")

    for _, aug_group in augmentables.items():
        if "images" in aug_group:
            images = _ensure_nxyc(aug_group["images"])  # type: ignore
        else:
            images = ref_images

        if mode == "2d":
            res = augmenter.augment(
                images=images,
                heatmaps=_ensure_nxyc(aug_group["heatmaps"])  # type: ignore
                if "heatmaps" in aug_group
                else None,
                segmentation_maps=_ensure_nxyc(aug_group["segmentation_maps"])  # type: ignore
                if "segmentation_maps" in aug_group
                else None,
                keypoints=aug_group.get("keypoints", None),
                bounding_boxes=aug_group.get("bounding_boxes", None),
                polygons=aug_group.get("polygons", None),
                line_strings=aug_group.get("line_strings", None),
                return_batch=True,
            )

        elif mode == "3d":
            # Process the stack one at a time while making sure the same transformation
            # is being applied. See https://github.com/aleju/imgaug/issues/51
            assert images is not None
            assert isinstance(images, np.ndarray | torch.Tensor)

            segmentation_maps = None
            if "segmentation_maps" in aug_group:
                assert isinstance(aug_group["segmentation_maps"], np.ndarray | torch.Tensor)
                segmentation_maps = _ensure_nxyc(aug_group["segmentation_maps"])
                # if segmentation_maps.dtype != np.uint8:
                #     raise ValueError(
                #         f"{key} is a `segmentation_maps` which is {segmentation_maps.dtype} "
                #         "but needs to be uint8"
                #     )
                #     # TODO: checking dtype would also be useful for 2d mode

            heatmaps = None
            if "heatmaps" in aug_group:
                assert isinstance(aug_group["heatmaps"], np.ndarray | torch.Tensor)
                heatmaps = _ensure_nxyc(aug_group["heatmaps"])

            ## Crop `ref_images` if necessary
            if (
                segmentation_maps is not None
                and images.shape[-4:-1] != segmentation_maps.shape[-4:-1]
            ):
                assert images is ref_images, "Possible bug if image is actually used"
                images = crop_center(images, segmentation_maps.shape[-4:-1] + (images.shape[-1],))
            if heatmaps is not None and images.shape[-4:-1] != heatmaps.shape[-4:-1]:
                assert images is ref_images, "Possible bug if image is actually used"
                images = crop_center(images, heatmaps.shape[-4:-1] + (images.shape[-1],))
            if heatmaps is not None and segmentation_maps is not None:
                assert segmentation_maps.shape[-4:-1] == heatmaps.shape[-4:-1]

            images_list = [x[np.newaxis, :] for x in images]
            seg_list: Sequence[None] | Sequence[np.ndarray | torch.Tensor]
            heat_list: Sequence[None] | Sequence[np.ndarray | torch.Tensor]
            if segmentation_maps is None:
                seg_list = [None for _ in images]
            else:
                seg_list = [x[np.newaxis, :] for x in segmentation_maps]
            if heatmaps is None:
                heat_list = [None for _ in images]
            else:
                heat_list = [x[np.newaxis, :] for x in heatmaps]

            res_images = []
            res_seg = []
            res_heat = []
            for i, s, h in zip(images_list, seg_list, heat_list):
                res_ = augmenter.augment(
                    images=i,
                    heatmaps=h,
                    segmentation_maps=s,
                    return_batch=True,
                )
                res_images.append(np.squeeze(res_.images_aug, axis=0))
                if s is not None:
                    res_seg.append(np.squeeze(res_.segmentation_maps_aug, axis=0))
                if h is not None:
                    res_heat.append(np.squeeze(res_.heatmaps_aug, axis=0))

            res = FakeResult(
                np.stack(res_images, axis=0) if res_images else None,
                np.stack(res_seg, axis=0) if res_seg else None,
                np.stack(res_heat, axis=0) if res_heat else None,
            )

        if "images" in aug_group:
            aug_group["images"] = _ensure_cxyn(res.images_aug, aug_group["images"])  # type: ignore
        if "heatmaps" in aug_group:
            aug_group["heatmaps"] = _ensure_cxyn(
                res.heatmaps_aug, aug_group["heatmaps"]  # type: ignore
            )
        if "segmentation_maps" in aug_group:
            aug_group["segmentation_maps"] = _ensure_cxyn(
                res.segmentation_maps_aug, aug_group["segmentation_maps"]  # type: ignore
            )
        if "keypoints" in aug_group:
            aug_group["keypoints"] = res.keypoints_aug
        if "bounding_boxes" in aug_group:
            aug_group["bounding_boxes"] = res.bounding_boxes_aug
        if "polygons" in aug_group:
            aug_group["polygons"] = res.polygons_aug
        if "line_strings" in aug_group:
            aug_group["line_strings"] = res.line_strings_aug

    return _ungroup_kwargs(augmentables, kwarg_mapping=kwarg_mapping)


for attr in dir(iaa):
    if attr[0].isupper() and hasattr(getattr(iaa, attr), "augment"):
        builder.register(f"imgaug.augmenters.{attr}")(getattr(iaa, attr))

for attr in dir(iaa.imgcorruptlike):
    if attr[0].isupper() and hasattr(getattr(iaa.imgcorruptlike, attr), "augment"):
        builder.register(f"imgaug.augmenters.imgcorruptlike.{attr}")(
            getattr(iaa.imgcorruptlike, attr)
        )
