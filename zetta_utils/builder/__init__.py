"""Building objects from specs"""
import inspect
from typing import List
import numpy as np

from . import built_in_registrations, constants
from .building import (
    SPECIAL_KEYS,
    BuilderPartial,
    UnpicklableDict,
    build,
    get_initial_builder_spec,
)
from .registry import REGISTRY, get_matching_entry, register, unregister

PARALLEL_BUILD_ALLOWED: bool = False

NP_MANUAL: List[str] = []
for k in dir(np):
    if not k.startswith("_") and k not in NP_MANUAL:
        if inspect.isroutine(getattr(np, k)):
            register(f"np.{k}")(getattr(np, k))
        elif isinstance(getattr(np, k), (float, type(None))):
            # Add inf, nan, newaxis (None), etc. to cue specs
            register(f"np.{k}")(lambda k=k: getattr(np, k))
