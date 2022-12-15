import codecs
import zlib

import dill


def serialize(obj):  # pragma: no cover
    return codecs.encode(zlib.compress(dill.dumps(obj, protocol=4)), "base64").decode()


def deserialize(s):  # pragma: no cover
    return dill.loads(zlib.decompress(codecs.decode(s.encode(), "base64")))
