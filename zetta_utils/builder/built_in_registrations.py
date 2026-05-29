from __future__ import annotations

import importlib
import inspect
import re
from typing import Any, Callable, Optional

from packaging.specifiers import SpecifierSet

from . import constants
from .building import BuilderPartial
from .registry import RegistryEntry, register, register_dynamic_resolver

# Aliases a lambda string may reference, mapped to their importable module name.
# Single source of truth shared with the forkserver preload computation so the
# template preloads exactly the libraries a spec's lambdas will use.
LAMBDA_LIBS: tuple[tuple[str, str], ...] = (("np", "numpy"), ("torch", "torch"))


def lambda_referenced_libs(lambda_str: str) -> list[tuple[str, str]]:
    """Return the (alias, module) pairs from `LAMBDA_LIBS` used in `lambda_str`."""
    return [(alias, mod) for alias, mod in LAMBDA_LIBS if re.search(rf"\b{alias}\b", lambda_str)]


@register("lambda", allow_partial=False)
def efficient_parse_lambda_str(lambda_str: str, name: Optional[str] = None) -> Callable:
    """Parses strings that are lambda functions"""
    if not isinstance(lambda_str, str):
        raise TypeError("`lambda_str` must be a string.")
    if not lambda_str.startswith("lambda"):
        raise ValueError("`lambda_str` must start with 'lambda'.")

    return BuilderPartial(spec={"@type": "invoke_lambda_str", "lambda_str": lambda_str}, name=name)


@register("invoke_lambda_str", allow_partial=False)
def invoke_lambda_str(*args: list, lambda_str: str, **kwargs: dict) -> Any:
    eval_globals = dict(globals())
    for alias, module_name in lambda_referenced_libs(lambda_str):
        eval_globals[alias] = importlib.import_module(module_name)
    return eval(lambda_str, eval_globals)(*args, **kwargs)  # pylint: disable=eval-used


_DEFAULT_SPEC = SpecifierSet(constants.DEFAULT_VERSION_SPEC)


def _resolve_numpy(name: str) -> RegistryEntry | None:
    """Resolve `np.<attr>` lookups by deferring the numpy import to first use."""
    # pylint: disable=import-outside-toplevel
    suffix = name[len("np.") :]
    if not suffix or suffix.startswith("_"):
        return None
    import numpy as np

    if not hasattr(np, suffix):
        return None
    attr = getattr(np, suffix)
    if inspect.isroutine(attr):
        fn: Callable = attr
    elif isinstance(attr, (float, type(None))):
        # Constants like np.inf / np.nan / np.newaxis: wrap in a thunk so the
        # builder treats them as zero-arg builders.
        def fn(suffix=suffix):
            return getattr(np, suffix)

    else:
        return None
    return RegistryEntry(
        fn=fn,
        allow_partial=True,
        allow_parallel=True,
        version_spec=_DEFAULT_SPEC,
    )


def _resolve_torch(name: str) -> RegistryEntry | None:
    """Resolve `torch.<dotted.path>` lookups by walking the torch namespace.

    Replaces the import-time loops over torch.nn / torch.optim / torch.linalg
    / torch.fft / torch.nn.functional that used to register every public
    routine/class up front. Any explicitly decorated `torch.*` name (e.g.
    `torch.nn.Sequential` defined in convnet/architecture/primitives.py) wins
    via the static-index fallback before the resolver is consulted.
    """
    # pylint: disable=import-outside-toplevel
    suffix = name[len("torch.") :]
    if not suffix:
        return None
    import torch

    obj: Any = torch
    for part in suffix.split("."):
        if not part or part.startswith("_") or not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    if not callable(obj):
        return None
    return RegistryEntry(
        fn=obj,
        allow_partial=True,
        allow_parallel=True,
        version_spec=_DEFAULT_SPEC,
    )


register_dynamic_resolver("np.", _resolve_numpy)
register_dynamic_resolver("torch.", _resolve_torch)
