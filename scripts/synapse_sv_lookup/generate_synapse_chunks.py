import math
import os
import multiprocessing

import numpy as np
import pandas as pd
from tqdm import tqdm

first_time = True
base_dir = "./murthy_fly"

if first_time:
    path_in = os.path.join(base_dir, "final_edgelist.df")
    with open(path_in, "r") as fi:
        lines = fi.readlines()
    data = np.ones([len(lines) - 1, 2, 3]).astype(np.int64) * -1
    cleft_ids = np.ones([len(lines) - 1]).astype(np.int64) * -1

    for i, line in enumerate(tqdm(lines[1:])):
        values = line.split(",")
        data[i][0] = [int(e) for e in values[12:15]]
        data[i][1] = [int(e) for e in values[15:18]]
        cleft_ids[i] = int(float(values[0]))
    with open(os.path.join(base_dir, "all_cleft_ids.npy"), "wb") as f:
        np.save(f, cleft_ids)

    with open(os.path.join(base_dir, "all_coords_vx.npy"), "wb") as f:
        np.save(f, data)
else:
    with open(os.path.join(base_dir, "all_cleft_ids.npy"), "rb") as f:
        cleft_ids = np.load(f)
    with open(os.path.join(base_dir, "all_coords_vx.npy"), "rb") as f:
        data = np.load(f)

chunk_size = [1024, 1024, 128]
chunk_ids = (data[:, 0] / chunk_size).astype(int)

work_dir = os.path.join(base_dir, f"./_chunks_{chunk_size[0]}_{chunk_size[1]}_{chunk_size[2]}")
chunk_ids_path = os.path.join(work_dir, "nonempty_chunk_ids.npy")
first_time = True
if first_time:
    try:
        os.mkdir(work_dir)
    except:
        ...
    n_partitions = 8
    partition_size = math.ceil(chunk_ids.shape[0] / n_partitions)
    nonempty_chunk_ids = np.ones([0, 3])
    for i in tqdm(range(n_partitions)):
        partition = chunk_ids[i * partition_size : (i + 1) * partition_size]
        partition_nonempties = np.unique(partition, axis=0)
        nonempty_chunk_ids = np.unique(
            np.concatenate([nonempty_chunk_ids, partition_nonempties], axis=0), axis=0
        )
        print(nonempty_chunk_ids.shape)
        # nonempty_chunk_ids = np.unique(chunk_ids, axis=1)
    with open(chunk_ids_path, "wb") as f:
        np.save(f, nonempty_chunk_ids)
else:
    with open(chunk_ids_path, "rb") as f:
        nonempty_chunk_ids = np.load(f)

def process_chunk(i):
    x, y, z = nonempty_chunk_ids[i]
    mask = (chunk_ids == np.array([x, y, z])).all(axis=1)
    if mask.sum() > 0:
        this_cleft_ids = cleft_ids[mask]
        this_coords_vx = data[mask]

        with open(f"{work_dir}/cleft_ids_{x}_{y}_{z}.npy", "wb") as f:
            np.save(f, this_cleft_ids)
        with open(f"{work_dir}/coords_vx_{x}_{y}_{z}.npy", "wb") as f:
            np.save(f, this_coords_vx)

tasks = list(range(nonempty_chunk_ids.shape[0]))
with multiprocessing.Pool(24) as pool:
    results = list(tqdm(pool.imap_unordered(process_chunk, tasks), total=len(tasks)))
