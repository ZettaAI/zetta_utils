import attrs
from zetta_utils.mazepa.flows import Dependency, flow_schema_cls


@flow_schema_cls
class Foo:
    def flow(self):
        yield Dependency


@flow_schema_cls
@attrs.mutable
class ComputeFieldFlowSchema:
    expand_bbox_resolution: bool = False
    expand_bbox_backend: bool = False
    expand_bbox_processing: bool = False
    shrink_processing_chunk: bool = False

    def flow(
        self,
    ):
        ...


breakpoint()
foo = ComputeFieldFlowSchema()
print(foo)
print(foo.flow())
