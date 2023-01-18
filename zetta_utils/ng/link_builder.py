import os
from typing import Literal, Optional, Sequence

import neuroglancer
import numpy as np
import requests
from typeguard import typechecked

from zetta_utils import builder, log
from zetta_utils.typing import Vec3D

logger = log.get_logger("zetta_utils")


def _get_seunglab_branch_json_state(
    state: neuroglancer.veiwer_state.VeiwerState,
    position: Optional[Vec3D] = None,
    scale_bar_nm: float = 20.0,
) -> dict:
    json_state = neuroglancer.url_state.to_json(state)
    for e in json_state["layers"]:
        e["source"] = e["source"][0]["url"]

    json_state["navigation"] = {"zoomFactor": scale_bar_nm / 100}
    if position is not None:
        json_state["navigation"]["pose"] = {"position": {"voxelCoordinates": list(position)}}
    return json_state


@typechecked
@builder.register("make_ng_link", cast_to_vec3d=["position"])
def make_ng_link(
    layers: Sequence[Sequence[str]],
    position: Optional[Vec3D] = None,
    scale_bar_nm: float = 20.0,
    layout: Literal["xy", "4panel"] = "xy",
    title: str = "zutils",
    state_server_url: Optional[str] = "https://api.zetta.ai/json",
):  # pragma: no cover # visualization related code
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        for layer_spec in layers:
            if len(layer_spec) != 3:
                raise ValueError("Each layer spec must be a tripple of stirngs")
            if layer_spec[1] == "image":
                layer_type = neuroglancer.ImageLayer
            elif layer_spec[1] == "segmentation":
                layer_type = neuroglancer.SegmentationLayer
            else:
                raise ValueError(f"Unsupported layer type {layer_spec[1]}")

            s.layers[layer_spec[0]] = layer_type(source=layer_spec[2])
        if state_server_url is not None:
            if "NG_STATE_SERVER_TOKEN" not in os.environ:
                raise RuntimeError(
                    "Missing 'NG_STATE_SERVER_TOKEN' env variable, which is "
                    "required to run link shortening with a state server"
                )
            token = os.environ["NG_STATE_SERVER_TOKEN"]
            json_state = _get_seunglab_branch_json_state(
                state=s,
                position=position,
                scale_bar_nm=scale_bar_nm,
            )
            response = requests.post(
                os.path.join(state_server_url, "post"),
                json=json_state,
                headers={"Authorization": f"token {token}"},
            )
            # remove both quotes
            state_url = response.text[1:-2]
            ng_url = f"https://neuromancer-seung-import.appspot.com/?json_url={state_url}"
        else:
            s.title = title
            s.layout = neuroglancer.viewer_state.DataPanelLayout(layout)
            if position is not None:
                s.position = np.array(position)
            if scale_bar_nm is not None:
                s.crossSectionScale = scale_bar_nm / (100 * 1000 * 1000 * 1000)

            ng_url = neuroglancer.url_state.to_url(
                s, prefix="https://neuroglancer-demo.appspot.com"
            )

    logger.info(f"NG URL: \n{ng_url}")
    return ng_url
