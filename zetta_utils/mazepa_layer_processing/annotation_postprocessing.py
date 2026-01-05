"""
Module containing task and flow definitions for annotation layer operations.
"""

from zetta_utils import builder, mazepa
from zetta_utils.layer.volumetric import VolumetricAnnotationLayer


@mazepa.taskable_operation
def post_process_annotation_layer_op(target: VolumetricAnnotationLayer):  # pragma: no cover
    target.backend.post_process()


@builder.register("post_process_annotation_layer_flow")
@mazepa.flow_schema
def post_process_annotation_layer_flow(target: VolumetricAnnotationLayer):  # pragma: no cover
    # As post-processing is inherently a single-process job, there's no need
    # to make or yield a task for it; we can just run it directly on the head node.
    target.backend.post_process()
