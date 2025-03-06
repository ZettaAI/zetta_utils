import math
import os

import fsspec
import numpy as np
import pandas as pd
from tqdm import tqdm

project = "murthy_fly"
chunk_size = [1024, 1024, 128]

output_chunks_path = (
    f"/home/akhilesh/opt/zetta_utils/scripts/synapse_sv_lookup/{project}/output"
)
final_output_path = (
    f"/home/akhilesh/opt/zetta_utils/scripts/synapse_sv_lookup/{project}/sql_supervoxels.csv"
)

chunk_str = "_".join([str(x) for x in chunk_size])
nonempty_chunk_ids_path = f"/home/akhilesh/opt/zetta_utils/scripts/synapse_sv_lookup/{project}/_chunks_{chunk_str}/nonempty_chunk_ids.npy"
with open(nonempty_chunk_ids_path, "rb") as f:
    nonempty_chunk_ids = np.load(f)

chunks = []
for i in tqdm(range(nonempty_chunk_ids.shape[0])):
    x, y, z = nonempty_chunk_ids[i]
    file_name = os.path.join(output_chunks_path, f"{x}_{y}_{z}.csv")
    with open(file_name) as f:
        for line in f.readlines():
            r = line.strip().split(",")
            r = f"{r[0]},{r[1]},,{r[2]},"
            chunks.append(r)

final_content = "\n".join(chunks)
with open(final_output_path, "w") as f:
    f.write(final_content)
