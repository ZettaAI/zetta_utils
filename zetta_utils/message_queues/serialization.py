from __future__ import annotations

import codecs
import pickle
import zlib

import dill


def serialize(obj):  # pragma: no cover
    try:
        result = _serialize(obj, pickle)
    except (pickle.PicklingError, TypeError):
        result = _serialize(obj, dill)

    return result


def _serialize(obj, module):  # pragma: no cover
    result = codecs.encode(zlib.compress(module.dumps(obj, protocol=4)), "base64").decode()
    return result


def _deserialize(s, module):  # pragma: no cover
    result = module.loads(zlib.decompress(codecs.decode(s.encode(), "base64")))
    return result


def deserialize(s):  # pragma: no cover
    try:
        result = _deserialize(s, pickle)
    except (ModuleNotFoundError, KeyError):
        result = _deserialize(s, dill)
    return result


def test(obj):  # pragma: no cover
    ser = serialize(obj)
    deser = deserialize(ser)
    return deser == obj
