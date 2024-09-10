import json

from tqdm import tqdm

chunk_size = 10 ** 6

edgelist_path = "/mnt/disks/sergiy-x0/code/synapse_postproc/banc/final_edgelist.df"
result_path = "/mnt/disks/sergiy-x0/code/synapse_postproc/banc/final_edgelist_with_sv.df"
sv_mapping_path = "/mnt/disks/sergiy-x0/code/synapse_postproc/banc/final_sv_mapping_x0.csv"
sv_mapping_dict_path = "/mnt/disks/sergiy-x0/code/synapse_postproc/banc/final_sv_mapping_x0.json"
first_run = False

if first_run:
    with open(sv_mapping_path, "r") as f:
        sv_mapping_lines = f.readlines()
    sv_mapping_lines[0] = sv_mapping_lines[0].replace("cleft_segid,presyn_sv_id,postsyn_sv_id", "")

    print(sv_mapping_lines[0])
    sv_mapping_dict = {}
    for line in tqdm(sv_mapping_lines):
        cleft_segid, presyn_sv_id, postsyn_sv_id = line.strip().split(",")
        sv_mapping_dict[cleft_segid] = f"{presyn_sv_id},{postsyn_sv_id}"
    with open(sv_mapping_dict_path, "w") as f:
        json.dump(sv_mapping_dict, f, indent=3)
else:
    with open(sv_mapping_dict_path, "r") as f:
        sv_mapping_dict = json.load(f)

with open(edgelist_path, "r") as f:
    edgelist_lines = f.readlines()

with open(result_path, "w") as f:
    f.write(edgelist_lines[0].strip() + ",presyn_sv_id,postsyn_sv_id\n")
    for line in tqdm(edgelist_lines[1:]):
        cleft_id = line.split(",")[0]
        f.write(line.strip() + sv_mapping_dict[str(int(float(cleft_id)))] + "\n")
