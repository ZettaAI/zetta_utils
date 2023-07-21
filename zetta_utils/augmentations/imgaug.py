from __future__ import annotations

from typing import Any, Final, Sequence, Sized, Tuple, TypeVar, overload

from imgaug import augmenters as iaa
from imgaug.augmenters.meta import Augmenter
from numpy.typing import NDArray

from zetta_utils import builder
from zetta_utils.tensor_ops import common, convert
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


def _ensure_cxyn(
    x: Tensor | Sequence[Tensor], ref: TensorTypeVar | Sequence[TensorTypeVar]
) -> TensorTypeVar | list[TensorTypeVar]:
    if isinstance(ref, Sequence):
        assert isinstance(x, Sequence)
        return [
            convert.astype(
                common.rearrange(v, pattern="X Y C -> C X Y 1"), reference=ref[i]  # type: ignore
            )
            for i, v in enumerate(x)
        ]
    else:
        return convert.astype(
            common.rearrange(x, pattern="N X Y C -> C X Y N"), reference=ref  # type: ignore
        )


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
    **kwargs,  # and the augmenters in the kwargs
):
    assert len(args) == 1
    augmenters = kwargs.pop("augmenters", None)
    assert augmenters is not None
    if isinstance(args[0], dict):
        return imgaug_augment(augmenters, **args[0], **kwargs)
    else:  # Tensor
        return imgaug_augment(augmenters, images=args[0], **kwargs)["images"]


def imgaug_augment(
    augmenters: Augmenter | Sequence[Augmenter],
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

    if not any("images" in group for group in augmentables.values()):
        raise ValueError("Expected at least one image in `images` or `kwargs`")

    for aug_group in augmentables.values():
        res = augmenter.augment(
            images=_ensure_nxyc(aug_group["images"]),  # type: ignore
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
