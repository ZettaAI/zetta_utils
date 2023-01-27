from __future__ import annotations

import codecs
import pickle
import zlib

from .exceptions import MazepaException


def serialize(obj):  # pragma: no cover
    result = codecs.encode(zlib.compress(pickle.dumps(obj, protocol=4)), "base64").decode()
    return result


def test(obj):  # pragma: no cover
    ser = serialize(obj)
    deser = deserialize(ser)
    return deser == obj


def deserialize(s):  # pragma: no cover
    try:
        result = pickle.loads(zlib.decompress(codecs.decode(s.encode(), "base64")))
    except MazepaException as e:
        raise e
    except Exception as e:
        e.args = (
            f"{e}\n This exception occured during desearilization, indicating a mismatch "
            "between serialization and deserialization environments. "
            "This is most likely caused by worker image being _not_ up to date, "
            "or scheduler and worker are running different python versions.",
        )
        raise e
    return result
