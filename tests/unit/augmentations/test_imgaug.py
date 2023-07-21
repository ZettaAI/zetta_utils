import numpy as np
import pytest
import torch
from imgaug import augmenters as iaa

import zetta_utils as zu
from zetta_utils.augmentations.imgaug import imgaug_augment

from ..helpers import assert_array_equal


def test_imgaug_basic_ndarray():
    image = np.random.randint(0, 255, (3, 128, 128, 2), dtype=np.uint8)
    segmap = np.random.randint(0, 2 ** 15, (1, 64, 64, 2), dtype=np.uint16)
    keypoints = [[(0.0, 0.0), (0.0, 128.0)], [(128.0, 128.0), (64.0, 64.0)]]
    bboxes = [[(0.0, 0.0, 128.0, 128.0)], [(64.0, 64.0, 128.0, 128.0)]]
    polygons = [[(0.0, 0.0), (0.0, 128.0), (128.0, 128.0)]], [
        [(128.0, 0.0), (128.0, 128.0), (0.0, 0.0)]
    ]
    line_strings = [[(0.0, 0.0), (0.0, 128.0), (128.0, 128.0)]], [
        [(128.0, 0.0), (128.0, 128.0), (0.0, 0.0)]
    ]

    aug = iaa.Rot90()
    augmented = imgaug_augment(
        augmenters=aug,
        images=image,
        segmentation_maps=segmap,
        keypoints=keypoints,
        bounding_boxes=bboxes,
        polygons=polygons,
        line_strings=line_strings,
    )

    assert augmented.keys() == {
        "images",
        "segmentation_maps",
        "keypoints",
        "bounding_boxes",
        "polygons",
        "line_strings",
    }
    assert augmented["images"].shape == (3, 128, 128, 2)
    assert augmented["segmentation_maps"].shape == (1, 64, 64, 2)
    assert_array_equal(augmented["images"], np.rot90(image, axes=(2, 1)))
    assert_array_equal(augmented["segmentation_maps"], np.rot90(segmap, axes=(2, 1)))
    assert augmented["keypoints"] == [[(128.0, 0.0), (0.0, 0.0)], [(0.0, 128.0), (64.0, 64.0)]]


def test_imgaug_basic_tensor():
    image = torch.randint(0, 255, (3, 128, 128, 2), dtype=torch.uint8)
    segmap = torch.randint(0, 2 ** 15, (1, 64, 64, 2), dtype=torch.int16)
    keypoints = [[(0.0, 0.0), (0.0, 128.0)], [(128.0, 128.0), (64.0, 64.0)]]

    aug = iaa.Rot90()
    augmented = imgaug_augment(
        augmenters=aug, images=image, segmentation_maps=segmap, keypoints=keypoints
    )

    assert augmented.keys() == {"images", "segmentation_maps", "keypoints"}
    assert augmented["images"].shape == (3, 128, 128, 2)
    assert augmented["segmentation_maps"].shape == (1, 64, 64, 2)
    assert_array_equal(augmented["images"], torch.rot90(image, dims=(2, 1)))
    assert_array_equal(augmented["segmentation_maps"], torch.rot90(segmap, dims=(2, 1)))
    assert augmented["keypoints"] == [[(128.0, 0.0), (0.0, 0.0)], [(0.0, 128.0), (64.0, 64.0)]]


def test_imgaug_basic_lists_ndarray():
    image = [np.random.randint(0, 255, (1, 128, 128, 1), dtype=np.uint8) for _ in range(2)]
    heatmap = [np.random.rand(3, 64, 64, 1).astype(np.float32) for _ in range(2)]

    aug = iaa.Fliplr()
    augmented = imgaug_augment(augmenters=[aug], images=image, heatmaps=heatmap)

    assert augmented.keys() == {"images", "heatmaps"}
    assert len(augmented["images"]) == 2
    assert len(augmented["heatmaps"]) == 2
    assert_array_equal(augmented["images"][0], np.flip(image[0], axis=2))
    assert_array_equal(augmented["images"][1], np.flip(image[1], axis=2))
    assert_array_equal(augmented["heatmaps"][0], np.flip(heatmap[0], axis=2))
    assert_array_equal(augmented["heatmaps"][1], np.flip(heatmap[1], axis=2))


def test_imgaug_basic_lists_tensor():
    image = [torch.randint(0, 255, (1, 128, 128, 1), dtype=torch.uint8) for _ in range(2)]
    heatmap = [torch.rand(3, 64, 64, 1, dtype=torch.float32) for _ in range(2)]

    aug = iaa.Fliplr()
    augmented = imgaug_augment(augmenters=[aug], images=image, heatmaps=heatmap)

    assert augmented.keys() == {"images", "heatmaps"}
    assert len(augmented["images"]) == 2
    assert len(augmented["heatmaps"]) == 2
    assert_array_equal(augmented["images"][0], torch.flip(image[0], dims=(2,)))
    assert_array_equal(augmented["images"][1], torch.flip(image[1], dims=(2,)))
    assert_array_equal(augmented["heatmaps"][0], torch.flip(heatmap[0], dims=(2,)))
    assert_array_equal(augmented["heatmaps"][1], torch.flip(heatmap[1], dims=(2,)))


def test_imgaug_custom_lists():
    image = [np.random.randint(0, 255, (1, 128, 128, 1), dtype=np.uint8) for _ in range(2)]
    seg = [np.random.randint(0, 2 ** 15, (1, 128, 128, 1), dtype=np.uint16) for _ in range(2)]
    aff = [np.random.rand(3, 64, 64, 1).astype(np.float32) for _ in range(2)]

    aug = iaa.Add(10)
    augmented = imgaug_augment(
        augmenters=aug,
        src_img=image[0],
        tgt_img=image[1],
        src_seg=seg[0],
        tgt_seg=seg[1],
        src_aff=aff[0],
        tgt_aff=aff[1],
    )

    assert augmented.keys() == {"src_img", "tgt_img", "src_seg", "tgt_seg", "src_aff", "tgt_aff"}
    assert augmented["src_img"].shape == (1, 128, 128, 1)
    assert augmented["tgt_img"].shape == (1, 128, 128, 1)
    assert augmented["src_seg"].shape == (1, 128, 128, 1)
    assert augmented["tgt_seg"].shape == (1, 128, 128, 1)
    assert augmented["src_aff"].shape == (3, 64, 64, 1)
    assert augmented["tgt_aff"].shape == (3, 64, 64, 1)

    assert_array_equal(augmented["src_img"], (image[0].clip(0, 245) + 10))


def test_imgaug_mixed_lists():
    image_group = np.random.randint(0, 255, (2, 64, 64, 10), dtype=np.uint8)
    another_image = np.random.randint(0, 255, (3, 1024, 1024, 1), dtype=np.uint8)

    aug = [iaa.Invert()]
    augmented = imgaug_augment(
        augmenters=aug,
        images=image_group,
        another_img=another_image,
    )

    assert augmented.keys() == {"images", "another_img"}
    assert augmented["images"].shape == (2, 64, 64, 10)
    assert augmented["another_img"].shape == (3, 1024, 1024, 1)
    assert_array_equal(augmented["images"], np.invert(image_group))
    assert_array_equal(augmented["another_img"], np.invert(another_image))


def test_imgaug_exceptions():
    seg = [np.random.randint(0, 2 ** 15, (1, 128, 128, 1), dtype=np.uint16) for _ in range(2)]
    aug = iaa.Invert()

    with pytest.raises(ValueError):
        imgaug_augment(aug, data_seg=seg)

    with pytest.raises(ValueError):
        imgaug_augment(aug, data_unknownsuffix=seg)


def test_imgaug_builder():
    zu.load_all_modules()  # pylint: disable=protected-access
    spec = zu.builder.build(
        spec={
            "@type": "imgaug_readproc",
            "@mode": "partial",
            "augmenters": [
                {
                    "@type": "imgaug.augmenters.Sequential",
                    "children": [
                        {"@type": "imgaug.augmenters.Add", "value": 0},
                        {"@type": "imgaug.augmenters.imgcorruptlike.DefocusBlur", "severity": 5},
                    ],
                },
            ],
        }
    )
    arr = np.zeros((1, 128, 128, 1), dtype=np.uint8)
    assert spec({"images": arr}).keys() == {"images"}
    assert_array_equal(spec(arr), arr)
