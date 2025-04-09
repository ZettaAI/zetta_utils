from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Any, DefaultDict

import torch
import typeguard

from zetta_utils import builder, log, mazepa, tensor_ops
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.db_layer import DBLayer
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.tensor_ops.normalization import apply_clahe

from ..alignment import online_finetuner
from ..alignment.base_encoder import BaseEncoder
from .lens_correction import LensCorrectionModel
from .registry import zs_to_tiles
from .tiles import open_image_from_gcs

logger = log.get_logger("zetta_utils")


@builder.register("match_tiles_flow")
@mazepa.flow_schema
def match_tiles_flow(  # pylint: disable=too-many-locals, invalid-name
    tile_registry: DBLayer,
    pair_registry: DBLayer,
    crop: int,
    ds_factor: int,
    max_disp: int,
    z_start: int,
    z_stop: int,
    mask_encoder_path: str,
    mask_threshold: int = 4,
    mask_keep_larger_proportion: float = 0.25,
    mask_plug_holes_proportion: float = 0.05,
    max_candidate_mse_ratio: float = 1.10,
    max_num_candidates: int = 81,
    max_num_finetune: int = 10,
    lens_correction_model: LensCorrectionModel | None = None,
):
    """
    Computes the rough montage offsets and stores them to a match registry.

    :param tile_registry: The input tile registry.
    :param pairs_registry: The registry to store the pairs in.
    :param crop: How much to crop tiles in XY, after downsampling, before matching.
    :param ds_factor: How much to downsample the tiles in XY.
    :param z_start: Z offset to start, inclusive.
    :param z_stop: Z offset to stop, exclusive.
    :param mask_encoder_path: The path to the encoder for masking / thresholding.
    :param mask_threshold: Mask parameter; see `image_to_encoding_to_mask`.
    :param mask_keep_larger_proportion: Mask parameter; see `image_to_encoding_to_mask`.
    :param mask_plug_holes_proportion: Mask parameter; see `image_to_encoding_to_mask`.
    :param max_candidate_mse_ratio: Match parameter; see `finetune_mse_offset`.
    :param max_num_candidates: Match parameter; see `finetune_mse_offset`.
    :param max_num_finetune: Match parameter; see `finetune_mse_offset`.
    :param lens_correction_model: The lens correction model to use.
    """
    pairs_to_match = []
    z_inds = []

    tiles = zs_to_tiles(tile_registry, z_start, z_stop)

    for tile_src, dict_src in tiles.items():
        for tile_tgt, dict_tgt in tiles.items():
            if dict_tgt["z_index"] != dict_src["z_index"]:
                continue
            if (
                dict_src["x_index"] == dict_tgt["x_index"]
                and dict_tgt["y_index"] == dict_src["y_index"] - 1
            ) or (
                dict_src["y_index"] == dict_tgt["y_index"]
                and dict_tgt["x_index"] == dict_src["x_index"] - 1
            ):
                pairs_to_match.append(
                    (
                        tile_src,
                        tile_tgt,
                        dict_src["x_offset"] - dict_tgt["x_offset"],
                        dict_src["y_offset"] - dict_tgt["y_offset"],
                    )
                )
                z_inds.append(dict_tgt["z_index"])

    # debug
    #    pairs = pairs[7000:7100]
    pairs_tasks = [
        match_tile_pair.make_task(
            *pair,
            ds_factor=ds_factor,
            max_disp=max_disp,
            crop=crop,
            mask_threshold=mask_threshold,
            mask_keep_larger_proportion=mask_keep_larger_proportion,
            mask_plug_holes_proportion=mask_plug_holes_proportion,
            max_candidate_mse_ratio=max_candidate_mse_ratio,
            max_num_candidates=max_num_candidates,
            max_num_finetune=max_num_finetune,
            mask_encoder_path=mask_encoder_path,
            lens_correction_model=lens_correction_model,
        )
        for pair in pairs_to_match
    ]
    yield pairs_tasks
    yield mazepa.Dependency()
    for task in pairs_tasks:
        assert task.outcome is not None
        assert task.outcome.return_value is not None

    pairs_result = [(*task.outcome.return_value, z) for task, z in zip(pairs_tasks, z_inds)]  # type: ignore # pylint: disable = line-too-long
    pairs = [f"{pair[6]}_{pair[0]}_{pair[1]}" for pair in pairs_result]
    datas = [
        {
            "src": pair[0],
            "tgt": pair[1],
            "x_offset": pair[2],
            "y_offset": pair[3],
            "best_mse": pair[4],
            "patch_std": pair[5],
            "z": pair[6],
        }
        for pair in pairs_result
    ]

    pair_registry[
        pairs,
        ("src", "tgt", "x_offset", "y_offset", "best_mse", "patch_std", "z"),
    ] = datas


@builder.register("elastic_tile_placement_flow")
@mazepa.flow_schema
def elastic_tile_placement_flow(  # pylint: disable= invalid-name
    tile_registry_in: DBLayer,
    tile_registry_out: DBLayer,
    pair_registry: DBLayer,
    std_filter: float,
    z_start: int,
    z_stop: int,
    min_x: int = 0,
    min_y: int = 0,
    mse_consensus_filter: float = 3.0,
):
    """
    Elastically places tiles based on pairs. Matches are initially filtered based on
    standard deviation, and dynamically rejected on consensus every 5,000 iterations
    if the loss is reasonably stable. Only the largest connected component is kept.

    :param tile_registry_in: The input tile registry.
    :param tile_registry_out: The output tile registry.
    :param pairs_registry: The input pair registry
    :param std_filter: Minimum standard deviation of the template patch for a
     match to be considered valid.
    :param z_start: Z offset to start, inclusive.
    :param z_stop: Z offset to stop, exclusive.
    :param min_x: Where to place the tile with the smallest X offset once the placement
    is complete; used to avoid negative offsets.
    :param min_y: Where to place the tile with the smallest Y offset once the placement
    is complete; used to avoid negative offsets.
    :param mse_consensus_filter: The multiplier to offset MSE to be used as threshold
    for dynamically rejecting matches.
    """
    tasks = [
        _elastic_offsets.make_task(
            tile_registry_in=tile_registry_in,
            tile_registry_out=tile_registry_out,
            pair_registry=pair_registry,
            std_filter=std_filter,
            z=z,
            min_x=min_x,
            min_y=min_y,
            mse_consensus_filter=mse_consensus_filter,
        )
        for z in range(z_start, z_stop)
    ]
    yield tasks
    yield mazepa.Dependency()
    for task in tasks:
        assert task.outcome is not None
        assert task.outcome.return_value is not None


def _find_largest_tile_island(pairs: list[dict[str, Any]]) -> set[str]:
    tiles_islands: list[set[str]] = []
    for pair in pairs:
        src_island = None
        tgt_island = None
        for tiles_island in tiles_islands:
            if pair["src"] in tiles_island:
                src_island = tiles_island
            if pair["tgt"] in tiles_island:
                tgt_island = tiles_island
        if src_island is None and tgt_island is None:
            tiles_islands.append(set([pair["src"], pair["tgt"]]))
        elif src_island is None:
            assert tgt_island is not None
            tgt_island.add(pair["src"])
        elif tgt_island is None:
            assert src_island is not None
            src_island.add(pair["tgt"])
        elif src_island != tgt_island:
            tiles_islands.remove(src_island)
            tiles_islands.remove(tgt_island)
            tiles_islands.append(src_island.union(tgt_island))
    tiles_islands.sort(key=len)
    return tiles_islands[-1]


@mazepa.taskable_operation
def _elastic_offsets(  # pylint: disable= too-many-locals, too-many-branches, too-many-statements, too-many-nested-blocks
    tile_registry_in: DBLayer,
    tile_registry_out: DBLayer,
    pair_registry: DBLayer,
    std_filter: float,
    z: int,
    min_x: int,
    min_y: int,
    mse_consensus_filter: float,
) -> int:
    """
    Elastically determine the offset of tiles.
    """
    tiles_z = zs_to_tiles(tile_registry_in, z, z + 1)

    tile_inds = {}
    src_inds = []
    tgt_inds = []
    tile_offsets = torch.zeros(len(tiles_z), 2, dtype=torch.float32)
    tile_match_counts: DefaultDict[str, int] = defaultdict(lambda: 0)

    # populate all tile offsets
    for i, (tile, dict_tile) in enumerate(tiles_z.items()):
        tile_inds[tile] = i
        tile_offsets[i, 0] = dict_tile["x_offset"]
        tile_offsets[i, 1] = dict_tile["y_offset"]
    pairs_z = pair_registry[
        pair_registry.query({"z": [z]}),
        ("src", "tgt", "x_offset", "y_offset", "best_mse", "patch_std", "z"),
    ]
    pairs_z_stds = list(filter(lambda pair: pair["patch_std"] >= std_filter, pairs_z))
    tiles_to_accept = _find_largest_tile_island(pairs_z_stds)
    logger.info(
        f"{z}: {len(tiles_to_accept)} / {len(tiles_z)} tiles initially accepted as connected"
        f" based on template std filter, which accepted {len(pairs_z_stds)} / {len(pairs_z)}"
        " matches."
    )

    pair_ignore_count = 0
    pairs_to_accept = []
    for pair in pairs_z_stds:
        if pair["src"] not in tiles_to_accept or pair["tgt"] not in tiles_to_accept:
            pair_ignore_count += 1
        else:
            pairs_to_accept.append(pair)

    logger.info(f"{pair_ignore_count} of {len(pairs_z_stds)} ignored from rejected tiles.")

    pair_offsets = torch.zeros((len(pairs_to_accept), 2), dtype=torch.float32)
    mse_consensus_removed = torch.zeros(len(pairs_to_accept), dtype=torch.bool)

    for i, pair in enumerate(pairs_to_accept):
        pair_offsets[i, 0] = pair["x_offset"]
        pair_offsets[i, 1] = pair["y_offset"]
        tile_match_counts[pair["src"]] += 1
        tile_match_counts[pair["tgt"]] += 1
        src_inds.append(tile_inds[pair["src"]])
        tgt_inds.append(tile_inds[pair["tgt"]])
    pair_offsets = pair_offsets.to("cuda")
    tile_offsets = tile_offsets.to("cuda")
    tile_offsets.requires_grad = True
    optimizer = torch.optim.Adam([tile_offsets], 0.5)
    last_loss = 1e12
    pairs_to_remove = []
    for i in range(500000):
        optimizer.zero_grad()
        offsets = tile_offsets[src_inds] - tile_offsets[tgt_inds]
        squared_errors = ((pair_offsets - offsets) ** 2).sum(1)
        loss = squared_errors.mean()
        mse = loss.item()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            logger.info(f"Iteration {i}, MSE: {mse}")
        if (loss / last_loss) > 1 - 1e-8 and (i + 1) % 1000 != 0:
            break
        if (loss / last_loss) > 1 - 1e-3 and (i + 1) % 5000 == 0:
            for k in (squared_errors > mse_consensus_filter * mse).nonzero().cpu():
                if not mse_consensus_removed[k]:
                    pair = pairs_to_accept[k]
                    tile_match_counts[pair["src"]] -= 1
                    tile_match_counts[pair["tgt"]] -= 1
                    if tile_match_counts[pair["src"]] == 0:
                        logger.info("tile removed due to last match being removed.")
                        tiles_to_accept.remove(pair["src"])
                    if tile_match_counts[pair["tgt"]] == 0:
                        logger.info("tile removed due to last match being removed.")
                        tiles_to_accept.remove(pair["tgt"])
                    pairs_to_remove.append(pair)
                    mse_consensus_removed[k] = True
            logger.info(
                f"{torch.sum(mse_consensus_removed).item()} total matches removed during"
                " relaxation due to the consensus filter."
            )
        last_loss = loss

    pairs_to_accept_final = [pair for pair in pairs_to_accept if pair not in pairs_to_remove]

    with torch.no_grad():
        tile_offsets[:, 0] += min_x - tile_offsets[:, 0].min()
        tile_offsets[:, 1] += min_y - tile_offsets[:, 1].min()
    tiles_to_accept_list = list(_find_largest_tile_island(pairs_to_accept_final))
    logger.info(f"{z}: {len(tiles_to_accept)} / {len(tiles_z)} tiles finally accepted.")

    datas = [
        {
            "x_offset": tile_offsets[tile_inds[tile], 0].item(),
            "y_offset": tile_offsets[tile_inds[tile], 1].item(),
            "z_offset": z,
            "x_index": tiles_z[tile]["x_index"],
            "y_index": tiles_z[tile]["y_index"],
            "z_index": tiles_z[tile]["z_index"],
            "x_size": tiles_z[tile]["x_size"],
            "y_size": tiles_z[tile]["y_size"],
            "x_res": tiles_z[tile]["x_res"],
            "y_res": tiles_z[tile]["y_res"],
            "z_res": tiles_z[tile]["z_res"],
        }
        for tile in tiles_to_accept
    ]

    tile_registry_out[
        tiles_to_accept_list,
        (
            "x_offset",
            "y_offset",
            "z_offset",
            "x_index",
            "y_index",
            "z_index",
            "x_size",
            "y_size",
            "x_res",
            "y_res",
            "z_res",
        ),
    ] = datas

    return len(tiles_to_accept)


# TODO: add logger, write to datastore, add finetuning parameters as fn parameters
def finetune_mse_offset(  # pylint: disable= too-many-locals, too-many-statements
    src: torch.Tensor,
    tgt: torch.Tensor,
    src_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
    x_offset: int,
    y_offset: int,
    max_disp: int,
    max_candidate_mse_ratio: float = 1.10,
    max_num_candidates: int = 81,
    max_num_finetune: int = 10,
) -> tuple[int, int, float, float]:
    """
    Attempts to find the optimum offset between two tensors that minimises the MSE
    within some displacement of the initially given offsets. The MSE is only calculated
    on the overlapping area (after masking) at each location. If there are alternate
    candidates within ``max_candidate_mse_ratio`` ratio of the optimum mse ratio based on
    translation only, then up to ``max_num_finetune`` candidates will be finetuned before
    deciding on the optimum offset.

    Returns (x_offset, y_offset, best_mse, src_patch_std_dev).

    The pair will be considered unmatchable if one of the four scenarios below happens:
    1) The source patch (after masking) of the initially given overlap has a
    standard deviation of 0.
    2) All of the candidate target patches (overlaps with translation) have standard
    deviations of 0s.
    3) The best MSE candidate has an MSE of 0.
    4) There are more than ``max_num_candidates`` within ``max_candidate_mse_ratio`` of
    the best candidate.
    If the pair is unmatchable, then the function returns (x_offset, y_offset, 1e12, 0)
    for easy filtering.

    :param src: Source tensor.
    :param tgt: Target tensor.
    :param src_mask: One-hot image mask of the source tensor.
    :param tgt_mask: One-hot image mask of the target tensor.
    :param x_offset: The initial X offset.
    :param y_offset: The initial Y offset.
    :param max_disp: Supremum norm of the displacement to search from the initial offset.
    :param max_num_candidates: Max acceptable number of candidates within the MSE ratio.
    :param max_candidate_mse_ratio: Max acceptable ratio of translation-only MSE for
        candidates to finetune, relative to the best candidate's translation-only MSE.

    """
    unmatchable_ret = (x_offset, y_offset, 1e12, 0)

    src = src.squeeze()
    tgt = tgt.squeeze()

    assert src.ndim == 2
    assert tgt.ndim == 2

    src_idx = VolumetricIndex.from_coords(
        start_coord=Vec3D(x_offset, y_offset, 0),
        end_coord=Vec3D(x_offset + src.shape[0], y_offset + src.shape[1], 1),
        resolution=Vec3D(1, 1, 1),
    )
    tgt_idx = VolumetricIndex.from_coords(
        start_coord=Vec3D(0, 0, 0),
        end_coord=Vec3D(tgt.shape[0], tgt.shape[1], 1),
        resolution=Vec3D(1, 1, 1),
    )
    src_masked = (src * src_mask).unsqueeze(-1).float()
    tgt_masked = (tgt * tgt_mask).unsqueeze(-1).float()

    template_idx, template_in_src = tgt_idx.get_intersection_and_subindex(src_idx)
    template = src_masked[template_in_src]
    std = template.std().item()
    if std == 0:
        logger.info("Tile pair is unmatchable: the template has zero standard deviation.")
        return unmatchable_ret

    # generate the list of offsets to check, and make VolumetricIndices out of them
    offsets = list(product(range(-max_disp, max_disp + 1), range(-max_disp, max_disp + 1)))
    offsets_to_ind = {offset: i for i, offset in enumerate(offsets)}
    offset_tgt_idxs = [template_idx.translated((offset[0], offset[1], 0)) for offset in offsets]

    # generate slices for where to look in the target tensor for each offset and where to place
    # in the targets tensor; slices are much faster than using memory-based VolumetricLayer +
    # VolumetricIndices.
    targets = torch.zeros(
        (template_idx.shape[0], template_idx.shape[1], len(offsets)), dtype=tgt_masked.dtype
    )
    with typeguard.suppress_type_checks():
        offset_tgts_in_whole = [
            offset_tgt_idx.get_intersection_and_subindex(tgt_idx)[1]
            for offset_tgt_idx in offset_tgt_idxs
        ]
        whole_in_offset_tgts = [
            (*tgt_idx.get_intersection_and_subindex(offset_tgt_idx)[1][0:2], slice(i, i + 1, None))
            for i, offset_tgt_idx in enumerate(offset_tgt_idxs)
        ]
    for i, offset in enumerate(offsets):
        targets[(whole_in_offset_tgts[i])] = tgt_masked[offset_tgts_in_whole[i]]

    if targets.std() == 0:
        logger.info("Tile pair is unmatchable: all possible targets have zero standard deviation.")
        return unmatchable_ret

    mse_mask = template * targets != 0
    mses = torch.sum((template - targets) ** 2 * mse_mask, (0, 1)) / torch.sum(mse_mask, (0, 1))
    offset_mses = {offset: mse.item() for offset, mse in zip(offsets, mses)}
    offsets.sort(key=lambda offset: offset_mses[offset])

    if offset_mses[offsets[0]] == 0:
        logger.info("Tile pair is unmatchable: best offset has zero loss.")
        return unmatchable_ret
    num_offsets = 0

    while offset_mses[offsets[num_offsets]] < offset_mses[offsets[0]] * max_candidate_mse_ratio:
        num_offsets += 1
        if num_offsets == len(offsets):
            break
    print(num_offsets)
    if num_offsets > max_num_candidates:
        logger.info(f"Tile pair is unmatchable: too many ({num_offsets}) candidates.")
        return unmatchable_ret

    if num_offsets == 1:
        offset = offsets[0]
        return x_offset + offset[0], y_offset + offset[1], offset_mses[offset], std

    else:
        offsets_to_finetune = offsets[0 : min(num_offsets, max_num_finetune)]
        targets = targets[:, :, [offsets_to_ind[offset] for offset in offsets_to_finetune]]

        template = template.unsqueeze(0)
        targets = targets.unsqueeze(0)

        fields = [
            online_finetuner.align_with_online_finetuner(
                template[:, :, :, :],
                targets[:, :, :, i : i + 1],
                sm=1e2,
                num_iter=100,
                src_zeros_sm_mult=1,
                tgt_zeros_sm_mult=1,
                lr=1e-5,
            )
            for i in range(len(offsets_to_finetune))
        ]

        best_ind = 0
        best_mse = 1e12
        for i, field in enumerate(fields):
            template_warped = tensor_ops.common.rearrange(field, pattern="C X Y 1 -> 1 C X Y")(
                template[:, :, :, 0].squeeze()
            )
            target = targets[:, :, :, i].squeeze()
            mse_mask = template_warped * target != 0
            mse = ((template_warped - target) ** 2 * mse_mask).sum() / mse_mask.sum()
            if best_mse > mse:
                best_mse = mse.item()
                best_ind = i
                best_template_warped = template_warped  # pylint: disable= unused-variable
                best_target = target  # pylint: disable= unused-variable

        offset = offsets[best_ind]
        """
        # VISUALISATION
        import matplotlib.pyplot as plt

        print(best_mse, offset, std)
        plt.imshow(best_target.float() + 128, cmap="Greys")
        plt.imshow(best_template_warped.float() + 128, cmap="Blues", alpha=0.5)
        plt.show()
        """

        return x_offset + offset[0], y_offset + offset[1], best_mse, std


def image_to_encoding_to_mask(
    img: torch.Tensor,
    encoder_path: str,
    threshold: int,
    keep_larger_proportion: float,
    plug_holes_proportion: float,
) -> torch.Tensor:
    """
    Applies an encoder to make a mask by thresholding on absolute value,
    keeping components above ``keep_larger_proportion`` and plugging holes smaller than
    ``plug_holes_proportion``.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mask_encoder = BaseEncoder(encoder_path, preserve_size=True)

    img_enc = mask_encoder(img.byte().to(device).unsqueeze(-1).unsqueeze(0))
    mask = tensor_ops.mask.filter_cc(
        torch.logical_not(
            tensor_ops.mask.filter_cc(
                tensor_ops.common.compare(torch.abs(img_enc), "lt", threshold),
                "keep_large",
                round(img_enc.nelement() * keep_larger_proportion),
            )
        ),
        "keep_large",
        round(img_enc.nelement() * plug_holes_proportion),
    ).squeeze()
    return mask


# TODO: Semaphores?
@mazepa.taskable_operation
def match_tile_pair(  # pylint: disable= too-many-locals
    src_path: str,
    tgt_path: str,
    x_offset: int,
    y_offset: int,
    ds_factor: int,
    max_disp: int,
    crop: int,
    mask_threshold: int,
    mask_keep_larger_proportion: float,
    mask_plug_holes_proportion: float,
    max_candidate_mse_ratio: float,
    max_num_candidates: int,
    max_num_finetune: int,
    mask_encoder_path: str | None = None,
    lens_correction_model: LensCorrectionModel | None = None,
) -> tuple[str, str, int, int, float, float]:
    """
    Matches the tile at ``src_path`` to the one at ``tgt_path`` using the finetune_mse_offset.
    Initial guess ``x_offset`` and ``y_offset`` are given as src-tgt at full resolution
    (and so is ``crop``), while the ``max_disp`` is given as the supremum after downsampling.
    Applies CLAHE to the tiles for better results.
    """
    x_offset_ds = x_offset // ds_factor
    y_offset_ds = y_offset // ds_factor

    src_ds = open_image_from_gcs(src_path, ds_factor, crop, lens_correction_model)
    tgt_ds = open_image_from_gcs(tgt_path, ds_factor, crop, lens_correction_model)

    if mask_encoder_path is not None:
        src_mask = image_to_encoding_to_mask(
            src_ds,
            mask_encoder_path,
            mask_threshold,
            mask_keep_larger_proportion,
            mask_plug_holes_proportion,
        )
        tgt_mask = image_to_encoding_to_mask(
            tgt_ds,
            mask_encoder_path,
            mask_threshold,
            mask_keep_larger_proportion,
            mask_plug_holes_proportion,
        )
    else:
        src_mask = torch.ones_like(src_ds)
        tgt_mask = torch.ones_like(tgt_ds)
    new_x_offset, new_y_offset, peak_ratio, std = finetune_mse_offset(
        apply_clahe(src_ds),
        apply_clahe(tgt_ds),
        src_mask.cpu(),
        tgt_mask.cpu(),
        x_offset_ds,
        y_offset_ds,
        max_disp,
        max_candidate_mse_ratio,
        max_num_candidates,
        max_num_finetune,
    )
    return (
        src_path,
        tgt_path,
        new_x_offset * ds_factor,
        new_y_offset * ds_factor,
        peak_ratio,
        std,
    )
