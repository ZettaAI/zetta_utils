import attrs

from zetta_utils import mazepa
from zetta_utils.typing import IntVec3D


@attrs.frozen
class Foo:
    v: IntVec3D


f = Foo(IntVec3D(1, 2, 3))
mazepa.serialization.deserialize(mazepa.serialization.serialize(f))
