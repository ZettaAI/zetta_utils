"""Static discovery of @register-decorated symbols via AST.

Builds a `name -> module` index without executing any code in the scanned files.
Used to drive semi-lazy preload: the parent process scans, computes the minimal
module set needed for a given spec, and only those modules are imported into the
forkserver template.
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import pathlib
import re
import time
from typing import Iterable

import attrs

logger = logging.getLogger(__name__)

_REGISTER_NEEDLE = re.compile(rb"\bregister\b")
_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]


@attrs.frozen
class IndexEntry:
    """One @register decoration discovered statically."""

    name: str
    module: str
    qualname: str
    versions: str
    allow_partial: bool
    allow_parallel: bool
    source_path: str
    lineno: int


@attrs.frozen
class ScanWarning:
    """A decoration the scanner could not fully resolve."""

    source_path: str
    lineno: int
    reason: str


@attrs.frozen
class ScanResult:
    entries: tuple[IndexEntry, ...]
    warnings: tuple[ScanWarning, ...]
    scanned_files: int
    candidate_files: int
    elapsed_s: float

    def by_name(self) -> dict[str, list[IndexEntry]]:
        out: dict[str, list[IndexEntry]] = {}
        for e in self.entries:
            out.setdefault(e.name, []).append(e)
        return out


def _module_for_path(path: pathlib.Path, package_root: pathlib.Path) -> str:
    """Translate a .py path under the package root to a dotted module name."""
    rel = path.resolve().relative_to(package_root.parent)
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_register_call(node: ast.expr) -> bool:
    """True iff the decorator Call's callable is `register` or `*.register`."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "register":
        return True
    if isinstance(func, ast.Name) and func.id == "register":
        return True
    return False


def _literal_or_none(node: ast.expr | None):
    if isinstance(node, ast.Constant):
        return node.value
    return None


def _extract_kwargs(call: ast.Call) -> tuple[dict[str, object], list[str]]:
    """Return (resolved_kwargs, unresolved_keyword_names)."""
    out: dict[str, object] = {}
    unresolved: list[str] = []
    for kw in call.keywords:
        if kw.arg is None:
            unresolved.append("**kwargs")
            continue
        val = _literal_or_none(kw.value)
        if val is None and not isinstance(kw.value, ast.Constant):
            unresolved.append(kw.arg)
        else:
            out[kw.arg] = val
    return out, unresolved


def _emit_entry(
    register_call: ast.Call,
    qualname: str,
    module: str,
    path: pathlib.Path,
) -> tuple[IndexEntry | None, ScanWarning | None]:
    """Build an IndexEntry from a `register(...)` Call node, if its first arg is a literal."""
    name_arg = register_call.args[0] if register_call.args else None
    name = _literal_or_none(name_arg)
    if not isinstance(name, str):
        return None, ScanWarning(
            source_path=str(path),
            lineno=register_call.lineno,
            reason=f"register() first arg is not a string literal at {qualname!r}",
        )
    kwargs, unresolved = _extract_kwargs(register_call)
    warning: ScanWarning | None = None
    if unresolved:
        warning = ScanWarning(
            source_path=str(path),
            lineno=register_call.lineno,
            reason=(f"register({name!r}) has non-literal kwargs {unresolved}; using defaults"),
        )
    entry = IndexEntry(
        name=name,
        module=module,
        qualname=qualname,
        versions=str(kwargs.get("versions", ">=0.0.0")),
        allow_partial=bool(kwargs.get("allow_partial", True)),
        allow_parallel=bool(kwargs.get("allow_parallel", True)),
        source_path=str(path),
        lineno=register_call.lineno,
    )
    return entry, warning


def _scan_decorators(
    tree: ast.AST, module: str, path: pathlib.Path
) -> tuple[list[IndexEntry], list[ScanWarning]]:
    """Pattern 1: @register("name") def foo(): ..."""
    entries: list[IndexEntry] = []
    warnings: list[ScanWarning] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for dec in node.decorator_list:
            if not _is_register_call(dec):
                continue
            assert isinstance(dec, ast.Call)
            entry, warning = _emit_entry(dec, node.name, module, path)
            if entry is not None:
                entries.append(entry)
            if warning is not None:
                warnings.append(warning)
    return entries, warnings


def _scan_direct_calls(
    tree: ast.AST, module: str, path: pathlib.Path
) -> tuple[list[IndexEntry], list[ScanWarning]]:
    """Pattern 2: register("name")(target) at module level.

    Used for classmethods, third-party callables, anything you can't decorate.
    """
    entries: list[IndexEntry] = []
    warnings: list[ScanWarning] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr):
            continue
        outer = node.value
        if not isinstance(outer, ast.Call):
            continue
        inner = outer.func
        if not _is_register_call(inner):
            continue
        assert isinstance(inner, ast.Call)
        target = outer.args[0] if outer.args else None
        qualname = _qualname_of(target) if target is not None else "<unknown>"
        entry, warning = _emit_entry(inner, qualname, module, path)
        if entry is not None:
            entries.append(entry)
        if warning is not None:
            warnings.append(warning)
    return entries, warnings


def _scan_file(
    path: pathlib.Path, source: str, package_root: pathlib.Path
) -> tuple[list[IndexEntry], list[ScanWarning]]:
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        return [], [
            ScanWarning(source_path=str(path), lineno=e.lineno or 0, reason=f"parse error: {e}")
        ]

    module = _module_for_path(path, package_root)
    e1, w1 = _scan_decorators(tree, module, path)
    e2, w2 = _scan_direct_calls(tree, module, path)
    return e1 + e2, w1 + w2


def _qualname_of(node: ast.expr) -> str:
    """Best-effort qualname for a register target expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_qualname_of(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _qualname_of(node.func) + "(...)"
    return "<expr>"


def _candidate_files(roots: Iterable[pathlib.Path]) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            try:
                with path.open("rb") as f:
                    head = f.read(64 * 1024)
                if _REGISTER_NEEDLE.search(head):
                    out.append(path)
                    continue
                if path.stat().st_size > 64 * 1024:
                    if _REGISTER_NEEDLE.search(path.read_bytes()):
                        out.append(path)
            except OSError:  # pragma: no cover - filesystem race
                continue
    return out


def _cache_key(paths: list[pathlib.Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=str):
        try:
            st = p.stat()
        except OSError:  # pragma: no cover - filesystem race
            continue
        h.update(str(p).encode())
        h.update(str(st.st_mtime_ns).encode())
        h.update(str(st.st_size).encode())
    return h.hexdigest()


def _cache_path() -> pathlib.Path:
    base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return pathlib.Path(base) / "zetta_utils" / "builder_index.json"


def _load_cache(key: str) -> ScanResult | None:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if data.get("key") != key:
        return None
    try:
        entries = tuple(IndexEntry(**e) for e in data["entries"])
        warnings = tuple(ScanWarning(**w) for w in data["warnings"])
    except TypeError:
        return None
    return ScanResult(
        entries=entries,
        warnings=warnings,
        scanned_files=data.get("scanned_files", 0),
        candidate_files=data.get("candidate_files", 0),
        elapsed_s=data.get("elapsed_s", 0.0),
    )


def _save_cache(key: str, result: ScanResult) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": key,
            "entries": [attrs.asdict(e) for e in result.entries],
            "warnings": [attrs.asdict(w) for w in result.warnings],
            "scanned_files": result.scanned_files,
            "candidate_files": result.candidate_files,
            "elapsed_s": result.elapsed_s,
        }
        path.write_text(json.dumps(payload))
    except OSError as e:
        logger.debug("builder index cache write failed: %s", e)


def scan(
    roots: Iterable[pathlib.Path] | None = None,
    use_cache: bool = True,
) -> ScanResult:
    """Statically discover all @register decorations under the given roots.

    :param roots: Directories to scan. Defaults to the zetta_utils package.
    :param use_cache: If True, consult the on-disk cache keyed on file mtimes.
    """
    if roots is None:
        roots = [_PACKAGE_ROOT]
    roots = [pathlib.Path(r).resolve() for r in roots]

    t0 = time.perf_counter()
    candidates = _candidate_files(roots)

    if use_cache:
        key = _cache_key(candidates)
        cached = _load_cache(key)
        if cached is not None:
            return cached
    else:
        key = ""

    entries: list[IndexEntry] = []
    warnings: list[ScanWarning] = []
    for path in candidates:
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as e:
            warnings.append(
                ScanWarning(source_path=str(path), lineno=0, reason=f"read error: {e}")
            )
            continue
        # Resolve which configured root contains this file so the dotted
        # module name comes out relative to the right package.
        package_root = _PACKAGE_ROOT
        for root in roots:
            try:
                path.resolve().relative_to(root.parent)
                package_root = root
                break
            except ValueError:
                continue
        e_list, w_list = _scan_file(path, source, package_root)
        entries.extend(e_list)
        warnings.extend(w_list)

    total_files = sum(1 for r in roots for _ in r.rglob("*.py"))
    result = ScanResult(
        entries=tuple(entries),
        warnings=tuple(warnings),
        scanned_files=total_files,
        candidate_files=len(candidates),
        elapsed_s=time.perf_counter() - t0,
    )

    if use_cache and key:
        _save_cache(key, result)
    return result


_INDEX: ScanResult | None = None


def get_index(refresh: bool = False) -> ScanResult:
    """Return the process-wide static registration index, scanning if needed."""
    global _INDEX  # pylint: disable=global-statement
    if _INDEX is None or refresh:
        _INDEX = scan(use_cache=not refresh)
    return _INDEX
