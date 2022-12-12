# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Any, Optional, Tuple

import attrs
from typeguard import typechecked


@typechecked
@attrs.frozen
class LayerSetIndex:  # pylint: disable=too-few-public-methods
    layer_selection: Optional[Tuple[str, ...]]
    layer_idx: Any
