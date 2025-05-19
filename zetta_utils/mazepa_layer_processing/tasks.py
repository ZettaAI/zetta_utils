"""
Module containing task and flow definitions for annotation layer operations.
"""

from zetta_utils import builder, mazepa

# from zetta_utils.layer.volumetric import VolumetricAnnotationLayer


@mazepa.taskable_operation
def post_process_annotation_layer_op(target):  # pragma: no cover
    # target: VolumetricAnnotationLayer
    target.backend.post_process()


@builder.register("post_process_annotation_layer_flow")
@mazepa.flow_schema
def post_process_annotation_layer_flow(target):  # pragma: no cover
    # target: VolumetricAnnotationLayer
    yield post_process_annotation_layer_op.make_task(target)
