import attrs
import torch

from zetta_utils.alignment import aced_relaxation

from ..helpers import assert_array_equal


@attrs.mutable
class GetAcedOffsetsConfig:
    max_dist: int
    tissue_mask: torch.Tensor
    misalignment_masks: dict[str, torch.Tensor]
    pairwise_fields: dict[str, torch.Tensor]
    pairwise_fields_inv: dict[str, torch.Tensor]
    expected_match_offsets: torch.Tensor
    expected_img_mask: torch.Tensor
    expected_aff_mask: torch.Tensor

    @staticmethod
    def get_standard(max_dist=1, xy=8, z=12):
        tissue_mask = torch.ones(1, xy, xy, z).bool()
        misalignment_masks = {
            str(-k): torch.zeros(1, xy, xy, z).bool() for k in range(1, max_dist + 1)
        }
        pairwise_fields = {str(-k): torch.zeros(2, xy, xy, z) for k in range(1, max_dist + 1)}
        pairwise_fields_inv = {str(-k): torch.zeros(2, xy, xy, z) for k in range(1, max_dist + 1)}
        expected_match_offsets = torch.ones(1, xy, xy, z)
        expected_match_offsets[:, :, :, 0] = 0
        expected_img_mask = torch.zeros(1, xy, xy, z)
        expected_aff_mask = torch.zeros(1, xy, xy, z)

        return GetAcedOffsetsConfig(
            max_dist=max_dist,
            tissue_mask=tissue_mask,
            misalignment_masks=misalignment_masks,
            pairwise_fields=pairwise_fields,
            pairwise_fields_inv=pairwise_fields_inv,
            expected_match_offsets=expected_match_offsets,
            expected_img_mask=expected_img_mask,
            expected_aff_mask=expected_aff_mask,
        )


def test_get_aced_match_offsets_trivial() -> None:
    config = GetAcedOffsetsConfig.get_standard()

    result = aced_relaxation.get_aced_match_offsets(
        tissue_mask=config.tissue_mask,
        misalignment_masks=config.misalignment_masks,
        pairwise_fields=config.pairwise_fields,
        pairwise_fields_inv=config.pairwise_fields_inv,
        max_dist=config.max_dist,
    )
    assert_array_equal(result["match_offsets"].cpu(), config.expected_match_offsets)
    assert_array_equal(result["img_mask"].cpu(), config.expected_img_mask)
    assert_array_equal(result["aff_mask"].cpu(), config.expected_aff_mask)


def test_get_aced_match_offsets_single_misalign() -> None:
    config = GetAcedOffsetsConfig.get_standard()
    config.misalignment_masks["-1"][0, 4:8, 4:8, 4] = True
    config.expected_aff_mask[0, 4:8, 4:8, 4] = True

    result = aced_relaxation.get_aced_match_offsets(
        tissue_mask=config.tissue_mask,
        misalignment_masks=config.misalignment_masks,
        pairwise_fields=config.pairwise_fields,
        pairwise_fields_inv=config.pairwise_fields_inv,
        max_dist=config.max_dist,
    )
    assert_array_equal(result["match_offsets"].cpu(), config.expected_match_offsets)
    assert_array_equal(result["img_mask"].cpu(), config.expected_img_mask)
    assert_array_equal(result["aff_mask"].cpu(), config.expected_aff_mask)


def test_get_aced_match_offsets_single_nontissue() -> None:
    config = GetAcedOffsetsConfig.get_standard()
    config.tissue_mask[0, 4:8, 4:8, 4] = False
    config.expected_match_offsets[0, 4:8, 4:8, 4:6] = 0
    config.expected_img_mask[0, 4:8, 4:8, 4] = True
    config.expected_aff_mask[0, 4:8, 4:8, 4:6] = True

    result = aced_relaxation.get_aced_match_offsets(
        tissue_mask=config.tissue_mask,
        misalignment_masks=config.misalignment_masks,
        pairwise_fields=config.pairwise_fields,
        pairwise_fields_inv=config.pairwise_fields_inv,
        max_dist=config.max_dist,
    )
    assert_array_equal(result["match_offsets"].cpu(), config.expected_match_offsets)
    assert_array_equal(result["img_mask"].cpu(), config.expected_img_mask)
    assert_array_equal(result["aff_mask"].cpu(), config.expected_aff_mask)


def test_get_aced_match_offsets_single_nontissue_maxdist2() -> None:
    config = GetAcedOffsetsConfig.get_standard(max_dist=2)
    config.tissue_mask[0, 4:8, 4:8, 4] = False
    config.expected_match_offsets[0, 4:8, 4:8, 4] = 0
    config.expected_match_offsets[0, 4:8, 4:8, 5] = 2
    config.expected_img_mask[0, 4:8, 4:8, 4] = True
    # config.expected_aff_mask[0, 4:8, 4:8, 4:6] = True

    result = aced_relaxation.get_aced_match_offsets(
        tissue_mask=config.tissue_mask,
        misalignment_masks=config.misalignment_masks,
        pairwise_fields=config.pairwise_fields,
        pairwise_fields_inv=config.pairwise_fields_inv,
        max_dist=config.max_dist,
    )
    assert_array_equal(result["match_offsets"].cpu(), config.expected_match_offsets)
    assert_array_equal(result["img_mask"].cpu(), config.expected_img_mask)
    assert_array_equal(result["aff_mask"].cpu(), config.expected_aff_mask)


def test_get_aced_match_offsets_single_nontissue_misd_maxdist2() -> None:
    config = GetAcedOffsetsConfig.get_standard(max_dist=2)

    config.tissue_mask[0, 4:8, 4:8, 4] = False
    config.misalignment_masks["-1"][0, 0:4, 0:4, 4] = True

    config.expected_match_offsets[0, 0:4, 0:4, 4] = 2
    config.expected_match_offsets[0, 4:8, 4:8, 4] = 0
    config.expected_match_offsets[0, 4:8, 4:8, 5] = 2

    config.expected_img_mask[0, 0:4, 0:4, 3] = True
    config.expected_img_mask[0, 4:8, 4:8, 4] = True

    result = aced_relaxation.get_aced_match_offsets(
        tissue_mask=config.tissue_mask,
        misalignment_masks=config.misalignment_masks,
        pairwise_fields=config.pairwise_fields,
        pairwise_fields_inv=config.pairwise_fields_inv,
        max_dist=config.max_dist,
    )
    assert_array_equal(result["match_offsets"].cpu(), config.expected_match_offsets)
    assert_array_equal(result["img_mask"].cpu(), config.expected_img_mask)
    assert_array_equal(result["aff_mask"].cpu(), config.expected_aff_mask)


def test_get_aced_match_offsets_double_nontissue_maxdist2() -> None:
    config = GetAcedOffsetsConfig.get_standard(max_dist=2)
    config.tissue_mask[0, :, :, 4] = False
    config.tissue_mask[0, 4:8, 4:8, 5] = False

    config.expected_match_offsets[0, :, :, 4] = 0
    config.expected_match_offsets[0, :, :, 5] = 2
    config.expected_match_offsets[0, 4:8, 4:8, 5] = 0
    config.expected_match_offsets[0, 4:8, 4:8, 6] = 0

    config.expected_img_mask[0, :, :, 4] = True
    config.expected_img_mask[0, 4:8, 4:8, 5] = True

    config.expected_aff_mask[0, 4:8, 4:8, 4] = True
    config.expected_aff_mask[0, 4:8, 4:8, 6] = True

    result = aced_relaxation.get_aced_match_offsets(
        tissue_mask=config.tissue_mask,
        misalignment_masks=config.misalignment_masks,
        pairwise_fields=config.pairwise_fields,
        pairwise_fields_inv=config.pairwise_fields_inv,
        max_dist=config.max_dist,
    )
    assert_array_equal(result["match_offsets"].cpu(), config.expected_match_offsets)
    assert_array_equal(result["img_mask"].cpu(), config.expected_img_mask)
    assert_array_equal(result["aff_mask"].cpu(), config.expected_aff_mask)


def test_get_aced_match_offsets_nontissue_misalign_maxdist2_misalign() -> None:
    config = GetAcedOffsetsConfig.get_standard(max_dist=2)

    config.tissue_mask[0, 4:8, 4:8, 4] = False

    config.misalignment_masks["-2"][0, 6:8, 6:8, 5] = True

    config.expected_match_offsets[0, 4:8, 4:8, 4] = 0
    config.expected_match_offsets[0, 4:8, 4:8, 5] = 2

    config.expected_img_mask[0, 4:8, 4:8, 4] = True

    config.expected_aff_mask[0, 6:8, 6:8, 4] = True
    config.expected_aff_mask[0, 6:8, 6:8, 5] = True

    result = aced_relaxation.get_aced_match_offsets(
        tissue_mask=config.tissue_mask,
        misalignment_masks=config.misalignment_masks,
        pairwise_fields=config.pairwise_fields,
        pairwise_fields_inv=config.pairwise_fields_inv,
        max_dist=config.max_dist,
    )
    assert_array_equal(result["match_offsets"].cpu(), config.expected_match_offsets)
    assert_array_equal(result["img_mask"].cpu(), config.expected_img_mask)
    assert_array_equal(result["aff_mask"].cpu(), config.expected_aff_mask)


def test_get_aced_match_offsets_double_nontissue_maxdist2_field() -> None:
    config = GetAcedOffsetsConfig.get_standard(max_dist=2)
    config.tissue_mask[0, :, :, 4] = False
    config.tissue_mask[0, 4:8, 4:8, 5] = False

    config.pairwise_fields["-2"][:, :, :, 5] = 1
    config.pairwise_fields_inv["-2"][:, :, :, 5] = -1

    config.expected_match_offsets[0, :, :, 4] = 0
    config.expected_match_offsets[0, :, :, 5] = 2
    config.expected_match_offsets[0, 4:8, 4:8, 5] = 0
    config.expected_match_offsets[0, 4:8, 4:8, 6] = 0

    config.expected_img_mask[0, :, :, 4] = True
    config.expected_img_mask[0, 4:8, 4:8, 5] = True

    config.expected_aff_mask[0, 4:8, 4:8, 4] = True
    config.expected_aff_mask[0, 4:8, 4:8, 6] = True

    result = aced_relaxation.get_aced_match_offsets(
        tissue_mask=config.tissue_mask,
        misalignment_masks=config.misalignment_masks,
        pairwise_fields=config.pairwise_fields,
        pairwise_fields_inv=config.pairwise_fields_inv,
        max_dist=config.max_dist,
    )
    assert_array_equal(result["match_offsets"].cpu(), config.expected_match_offsets)
    assert_array_equal(result["img_mask"].cpu(), config.expected_img_mask)
    assert_array_equal(result["aff_mask"].cpu(), config.expected_aff_mask)
