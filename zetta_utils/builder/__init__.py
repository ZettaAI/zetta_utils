"""Building objects from specs"""
from . import built_in_registrations, constants
from .building import (
    SPECIAL_KEYS,
    BuilderPartial,
    UnpicklableDict,
    build,
    get_initial_builder_spec,
)
from .registry import (
    REGISTRY,
    get_matching_entry,
    register,
    register_dynamic_resolver,
    unregister,
)

PARALLEL_BUILD_ALLOWED: bool = False
