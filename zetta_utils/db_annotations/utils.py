from zetta_utils.db_annotations.annotation import delete_annotations, read_annotations
from zetta_utils.db_annotations.collection import delete_collections
from zetta_utils.db_annotations.layer_group import (
    delete_layer_groups,
    read_layer_groups,
)


def cascade_delete_collections(
    collection_ids: list[str],
) -> None:  # pragma: no cover # emulator doesn't support composite filter
    layer_groups = read_layer_groups(collection_ids=collection_ids)
    annotations = read_annotations(collection_ids=collection_ids)
    delete_layer_groups(list(layer_groups.keys()))
    delete_annotations(list(annotations.keys()))
    delete_collections(collection_ids)
