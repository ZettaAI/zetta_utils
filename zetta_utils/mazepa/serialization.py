from __future__ import annotations

import codecs
import zlib

import dill

from .exceptions import MazepaException


def serialize(obj):  # pragma: no cover
    return codecs.encode(zlib.compress(dill.dumps(obj, protocol=4)), "base64").decode()


def deserialize(s):  # pragma: no cover
    try:
        result = dill.loads(zlib.decompress(codecs.decode(s.encode(), "base64")))
    except MazepaException as e:
        raise e
    except e:
        raise RuntimeError(
            "Encountered an error during desearilization, indicating a mismatch "
            "between serialization and deserialization environments. "
            "This is most likely caused by worker image being _not_ up to date, "
            "or scheduler and worker are running different python versions."
        ) from e
    return result
