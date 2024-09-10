"""Building objects from specs"""
import inspect
from typing import List
import numpy as np
from . import constants
from .registry import REGISTRY, register, get_matching_entry, unregister
from .building import (
    SPECIAL_KEYS,
    build,
    BuilderPartial,
    get_initial_builder_spec,
    UnpicklableDict,
)
from . import built_in_registrations

PARALLEL_BUILD_ALLOWED: bool = False

NP_MANUAL: List[str] = []
for k in dir(np):
    if not k.startswith("_") and k not in NP_MANUAL and inspect.isroutine(getattr(np, k)):
        register(f"np.{k}")(getattr(np, k))
