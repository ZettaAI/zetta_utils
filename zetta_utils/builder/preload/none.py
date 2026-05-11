# pylint: disable=unused-import
"""Minimal preload: only the always-eager set.

Used by `setup_environment("none")` to skip eager loading of registration-only
modules. Workers fork from a template containing just the modules with
non-registration side effects; everything else is loaded on-demand via the
lookup-miss fallback in registry.get_matching_entry.
"""
from zetta_utils import builder, log, parallel  # noqa: F401
