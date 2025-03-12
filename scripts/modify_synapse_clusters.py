"""
This script takes a segmentation volume, and a NG state with several layers of line annotations:
    "TP": True positives: these clusters will be copied to the output volume unchanged
    "FP": False positives: these clusters will be deleted
    "FN": False negatives: these clusters will be added to the output volume
This script then generates a new image volume with clusters as described above.
"""

import sys

import nglui.parser
from caveclient import CAVEclient

cave_client: CAVEclient | None = None


def verify_cave_auth() -> bool:
    global cave_client

    cave_client = CAVEclient(datastack_name=None, server_address="https://global.daf-apis.com")

    try:
        cave_client.state
        return True  # no exception?  All's good!
    except:
        pass
    print("Authentication needed.")
    print("Go to: https://global.daf-apis.com/auth/api/v1/create_token")
    token = input("Enter token: ")
    cave_client.auth.save_token(token=token, write_to_server_file=True)
    return True


def get_layer_name_of_type(state, layer_type: str, prompt: str) -> str:
    names = nglui.parser.layer_names(state)
    numToName = {}
    for i, name in enumerate(names, start=1):
        layer = nglui.parser.get_layer(state, name)
        # print(f"{i}: {name} ({layer['type']})")
        if layer["type"] == layer_type:
            numToName[i] = name
    if len(numToName) == 0:
        print(f"No {layer_type} layers found in this state.")
        sys.exit()
    elif len(numToName) == 1:
        _, name = numToName.popitem()
        print(f"[{name}]")
        return name
    while True:
        for i, name in numToName.items():
            print(f"{i}. {name}")
        choice = input(f"Enter layer name or number ({prompt}): ")
        if choice in names:
            return choice
        ichoice = int(choice)
        if ichoice in numToName:
            return numToName[ichoice]


def get_annotation_layer_name(state, prompt: str):
    return get_layer_name_of_type(state, "annotation", prompt)


def get_segmentation_layer_name(state, prompt: str):
    return get_layer_name_of_type(state, "segmentation", prompt)


def main():
    verify_cave_auth()
    assert cave_client is not None
    print(f"Enter Neuroglancer state link or ID:")
    source_path = input("> ")
    state_id = source_path.split("/")[-1]  # in case full URL was given
    state = cave_client.state.get_state_json(state_id)
    seg_layer_name = get_segmentation_layer_name(state, "Synapse Segmentation")
    layer_data = nglui.parser.get_layer(state, seg_layer_name)
    source_path = layer_data["source"]
    print(f"Synapse segmentation path: {source_path}")


if __name__ == "__main__":
    main()
