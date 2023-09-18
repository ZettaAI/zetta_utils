from __future__ import annotations

import json
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from _typeshed import SupportsRead, SupportsWrite  # pragma: no cover


def _mark_python_types(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return {"__tuple__": [_mark_python_types(e) for e in obj]}
    if isinstance(obj, list):
        return [_mark_python_types(e) for e in obj]
    if isinstance(obj, dict):
        return {key: _mark_python_types(value) for key, value in obj.items()}
    else:
        return obj


class ZettaSpecJSONEncoder(json.JSONEncoder):
    def encode(self, o: Any) -> str:
        return super().encode(_mark_python_types(o))

    def iterencode(self, o: Any, _one_shot: bool = False) -> Iterator[str]:
        return super().iterencode(_mark_python_types(o), _one_shot=_one_shot)


def tuple_hook(obj):
    if "__tuple__" in obj:
        return tuple(obj["__tuple__"])
    else:
        return obj


def dumps(obj, **kwargs) -> str:
    return json.dumps(obj, cls=ZettaSpecJSONEncoder, **kwargs)


def dump(obj: Any, fp: SupportsWrite[str], **kwargs) -> None:
    json.dump(obj, fp, cls=ZettaSpecJSONEncoder, **kwargs)


def loads(s: str | bytes | bytearray, **kwargs) -> Any:
    return json.loads(s, object_hook=tuple_hook, **kwargs)


def load(fp: SupportsRead[str | bytes], **kwargs) -> Any:
    return json.load(fp, object_hook=tuple_hook, **kwargs)
