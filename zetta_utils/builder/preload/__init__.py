"""Preload module groups for the forkserver template.

Each submodule here imports a wave of zetta_utils submodules; importing the
submodule has the side effect of populating the builder REGISTRY (via
@register decorators) before forkserver children fork from the daemon.

ALWAYS_EAGER is the explicit minimum: modules with non-registration side
effects (forkserver patch, logging, builder registry itself, numpy/lambda
auto-registrations) that must be loaded regardless of preload mode. Adding to
this set requires a one-line comment explaining the side effect, since the
default expectation is that a module's only import-time effect is registration.
"""
from __future__ import annotations

import logging

ALWAYS_EAGER: tuple[str, ...] = (
    "zetta_utils.parallel",  # patches multiprocessing.forkserver
    "zetta_utils.log",  # logging configuration
    "zetta_utils.builder",  # REGISTRY + numpy/lambda auto-reg
)

logger = logging.getLogger(__name__)


def compute_preload_set(type_names: set[str] | list[str]) -> list[str]:
    """Resolve @type names to module paths via the static registry index.

    Returns a deterministically ordered list: ALWAYS_EAGER first (preserving
    declared order), then the resolved module paths sorted alphabetically.
    Names that the index doesn't recognise are logged at DEBUG and skipped —
    they'll be handled by the lookup-miss fallback in registry.get_matching_entry
    at lookup time.

    :param type_names: Iterable of @type values extracted from a spec.
    :return: Module paths to feed to multiprocessing.set_forkserver_preload().
    """
    # pylint: disable=import-outside-toplevel
    from zetta_utils.builder.scan import get_index

    index_by_name = get_index().by_name()
    resolved: set[str] = set()
    unresolved: list[str] = []

    for name in type_names:
        entries = index_by_name.get(name)
        if not entries:
            unresolved.append(name)
            continue
        for entry in entries:
            resolved.add(entry.module)

    if unresolved:
        logger.debug(
            "compute_preload_set: %d @type name(s) not in static index, "
            "deferring to lookup-miss fallback: %s",
            len(unresolved),
            sorted(unresolved)[:10],
        )

    # ALWAYS_EAGER is also imported by preload.none, but listing it explicitly
    # here documents intent for direct callers of set_forkserver_preload().
    eager_set = set(ALWAYS_EAGER)
    extras = sorted(resolved - eager_set)
    return list(ALWAYS_EAGER) + extras
