import os
from typing import Literal, Optional, Sequence

import neuroglancer
import numpy as np
import requests
from typeguard import typechecked

from zetta_utils import builder, log
from zetta_utils.geometry import Vec3D

logger = log.get_logger("zetta_utils")


def _get_seunglab_branch_json_state(
    state: neuroglancer.viewer_state.ViewerState,
    position: Optional[Vec3D] = None,
    scale_bar_nm: float = 20.0,
) -> dict:  # pragma: no cover # visualization related code
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
) -> Optional[str]:  # pragma: no cover # visualization related code
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        for layer_spec in layers:
            if len(layer_spec) != 3:
                raise ValueError("Each layer spec must be a tripple of stirngs")

            s.layers[layer_spec[0]] = neuroglancer.viewer_state.layer_types[layer_spec[1]](
                source=layer_spec[2]
            )
        s.layout = neuroglancer.viewer_state.DataPanelLayout(layout)

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
            if not response.ok:
                logger.warning(
                    f"Received status {response.status_code} response from the state "
                    f"server: '{response.text}'. Not making link."
                )
                ng_url = None
            else:
                # remove both quotes
                state_url = response.text[1:-2]
                ng_url = f"https://neuromancer-seung-import.appspot.com/?json_url={state_url}"
        else:
            s.title = title
            if position is not None:
                s.position = np.array(position)
            if scale_bar_nm is not None:
                s.crossSectionScale = scale_bar_nm / (100 * 1000 * 1000 * 1000)

            ng_url = neuroglancer.url_state.to_url(
                s, prefix="https://neuroglancer-demo.appspot.com"
            )

    logger.info(f"NG URL: \n{ng_url}")
    return ng_url
