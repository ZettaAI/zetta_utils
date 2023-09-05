# type: ignore
# pylint: skip-file
from __future__ import annotations

from cloudvolume import CloudVolume

from zetta_utils.api.v0 import *

SOURCE_PATHS = {
    # "microns_pinky": {"contiguous": True},
    # "microns_basil": {"contiguous": True},
    # "microns_minnie": {"contiguous": False},
    # "microns_interneuron": {"contiguous": False},
    # "aibs_v1dd": {"contiguous": False},
    "kim_n2da": {"contiguous": True},
    # "kim_pfc2022": {"contiguous": True},
    # "kronauer_cra9": {"contiguous": True},
    # "kubota_001": {"contiguous": True},
    # "lee_fanc": {"contiguous": False},
    # "lee_banc": {"contiguous": False},
    # "lee_ppc": {"contiguous": True},
    # "lee_mosquito": {"contiguous": False},
    # "lichtman_zebrafish": {"contiguous": False},
    # "prieto_godino_larva": {"contiguous": True},
    # "fafb_v15": {"contiguous": False},
    # "lichtman_h01": {"contiguous": False},
    # "janelia_hemibrain": {"contiguous": True},
    # "janelia_manc": {"contiguous": False},
    # "nguyen_thomas_2022": {"contiguous": True},
    "mulcahy_2022_16h": {"contiguous": True},
    # "wildenberg_2021_vta_dat12a": {"contiguous": True},
    "bumbarber_2013": {"contiguous": True},
    # "wilson_2019_p3": {"contiguous": True},
    # "ishibashi_2021_em1": {"contiguous": True},
    # "ishibashi_2021_em2": {"contiguous": True},
    # "templier_2019_wafer1": {"contiguous": True},
    # "templier_2019_wafer3": {"contiguous": True},
    # "lichtman_octopus2022": {"contiguous": True},
}

BASE_PATH = "gs://zetta-research-nico/encoder/"

for k, v in SOURCE_PATHS.items():
    misd_mask_thr_path = BASE_PATH + "pairwise_aligned/" + k + "/misd_mask_thr"
    annotation_layer_name = "zetta-research-nico/encoder/pairwise_aligned/" + k
    cv = CloudVolume(misd_mask_thr_path, progress=False, fill_missing=True)
    resolution = cv.resolution.tolist()

    data = cv[:, :, :].squeeze(-1) == 0
    if not v["contiguous"]:
        data[:, :, 1::2] = False

    valid_chunks = data.nonzero()
    annotations = [
        Vec3D(resolution[0] * (x + 0.5), resolution[1] * (y + 0.5), resolution[2] * z)
        for (x, y, z) in zip(*valid_chunks)
    ]
    print(f"Writing {len(annotations)} annotations for layer {k}")
    write_remote_annotations(annotation_layer_name, resolution, annotations)


# Writing 5019 annotations for layer microns_pinky
# Writing 2591 annotations for layer microns_basil
# Writing 2882 annotations for layer microns_minnie
# Writing 6923 annotations for layer microns_interneuron
# Writing 5805 annotations for layer aibs_v1dd
# Writing 446 annotations for layer kim_n2da
# Writing 3699 annotations for layer kim_pfc2022
# Writing 740 annotations for layer kronauer_cra9
# Writing 4744 annotations for layer kubota_001
# Writing 1605 annotations for layer lee_fanc
# Writing 742 annotations for layer lee_banc
# Writing 7219 annotations for layer lee_ppc
# Writing 1964 annotations for layer lee_mosquito
# Writing 2799 annotations for layer lichtman_zebrafish
# Writing 4584 annotations for layer prieto_godino_larva
# Writing 1795 annotations for layer fafb_v15
# Writing 6624 annotations for layer lichtman_h01
# Writing 5304 annotations for layer janelia_hemibrain
# Writing 2398 annotations for layer janelia_manc
# Writing 1847 annotations for layer nguyen_thomas_2022
# Writing 3379 annotations for layer mulcahy_2022_16h
# Writing 1704 annotations for layer wildenberg_2021_vta_dat12a
# Writing 7325 annotations for layer bumbarber_2013
# Writing 2092 annotations for layer wilson_2019_p3
# Writing 141 annotations for layer ishibashi_2021_em1
# Writing 166 annotations for layer ishibashi_2021_em2
# Writing 5401 annotations for layer templier_2019_wafer1
# Writing 3577 annotations for layer templier_2019_wafer3
# Writing 5673 annotations for layer lichtman_octopus2022
