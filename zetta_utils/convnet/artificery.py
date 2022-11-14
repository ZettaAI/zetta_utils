import artificery

from zetta_utils import builder

art = artificery.Artificery()


@builder.register("ArtificerySpec")
def parse_artificery(spec):  # pragma: no cover
    result = art.create_net(spec)
    return result
