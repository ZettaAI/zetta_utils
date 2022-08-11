import artificery  # type: ignore
from zetta_utils import builder

art = artificery.Artificery()


@builder.register("parse_artificery")
def parse_artificery(path):  # pragma: no cover
    result = art.parse(path)
    return result
