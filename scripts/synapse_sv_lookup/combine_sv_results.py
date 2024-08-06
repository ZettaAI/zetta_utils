import math
import os

import fsspec
import numpy as np
import pandas as pd
from tqdm import tqdm

output_chunks_path = (
    "/mnt/disks/sergiy-x0/code/zetta_utils/notebooks/sergiy/2048_2048_256/results_x0"
)
final_output_path = (
    "/mnt/disks/sergiy-x0/code/zetta_utils/notebooks/sergiy/2048_2048_256/final_sf_mapping_x0.csv"
)
# files = fsspec.filesystem('gs').ls(output_path)
nonempty_chunk_ids_path = "/mnt/disks/sergiy-x0/code/zetta_utils/notebooks/sergiy/synapse_chunks_2048_2048_256/nonempty_chunk_ids.npy"
with open(nonempty_chunk_ids_path, "rb") as f:
    nonempty_chunk_ids = np.load(f)

chunks = []
for i in tqdm(range(nonempty_chunk_ids.shape[0])):
    x, y, z = nonempty_chunk_ids[i]
    file_name = os.path.join(output_chunks_path, f"{x}_{y}_{z}.csv")
    with open(file_name) as f:
        chunks.append(f.read())

final_content = "cleft_segid,presyn_sv_id,postsyn_sv_id" + "\n".join(chunks)
with open(final_output_path, "w") as f:
    f.write(final_content)
