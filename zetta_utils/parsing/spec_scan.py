"""Walk a parsed spec tree to collect builder @type / @version references.

Used by setup_environment('auto') to compute a minimal forkserver preload set
from a CUE file: the only modules that get eagerly loaded are those whose
@register name appears as an @type literal in the spec.
"""
from __future__ import annotations

import attrs

TYPE_KEY = "@type"
VERSION_KEY = "@version"
LAMBDA_STR_KEY = "lambda_str"


@attrs.frozen
class TypeRef:
    name: str
    version: str | None  # None means "use default"


@attrs.frozen
class SpecScanResult:
    types: tuple[TypeRef, ...]
    has_dynamic_types: bool  # True iff any @type value is non-string
    lambda_strs: tuple[str, ...] = ()

    def names(self) -> set[str]:
        return {t.name for t in self.types}


def extract_types(spec) -> SpecScanResult:
    """Walk an arbitrary parsed-spec value and collect every @type literal.

    Recurses through dicts, lists, and tuples. Records `has_dynamic_types`
    when an @type value isn't a plain string — those callers will need to
    fall back to eager preload. Also collects `lambda_str` string values: the
    modules a lambda uses are invisible to @type resolution, so the preload
    computation inspects these to keep the forkserver template complete.
    """
    types: list[TypeRef] = []
    lambda_strs: list[str] = []
    has_dynamic = False

    def _walk(node) -> None:
        nonlocal has_dynamic
        if isinstance(node, dict):
            tname = node.get(TYPE_KEY)
            if tname is not None:
                if isinstance(tname, str):
                    version = node.get(VERSION_KEY)
                    types.append(
                        TypeRef(
                            name=tname,
                            version=version if isinstance(version, str) else None,
                        )
                    )
                else:
                    has_dynamic = True
            lambda_str = node.get(LAMBDA_STR_KEY)
            if isinstance(lambda_str, str):
                lambda_strs.append(lambda_str)
            for k, v in node.items():
                if k in (TYPE_KEY, VERSION_KEY):
                    continue
                _walk(v)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _walk(item)

    _walk(spec)
    return SpecScanResult(
        types=tuple(types),
        has_dynamic_types=has_dynamic,
        lambda_strs=tuple(lambda_strs),
    )
