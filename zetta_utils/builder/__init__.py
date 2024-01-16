"""Building objects from specs"""
from . import constants
from .registry import REGISTRY, register, get_matching_entry, unregister
from .build import SPECIAL_KEYS, build, BuilderPartial, get_initial_builder_spec
from . import built_in_registrations

PARALLEL_BUILD_ALLOWED: bool = False
