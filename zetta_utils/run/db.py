from zetta_utils import constants
from zetta_utils.layer.db_layer.firestore import build_firestore_layer

RUN_COLLECTION = "run-info"
NODE_COLLECTION = "node-info"
GCS_STATS_COLLECTION = "gcs-stats-proxy"


NODE_DB = build_firestore_layer(
    NODE_COLLECTION, database=constants.RUN_DATABASE, project=constants.DEFAULT_PROJECT
)

RUN_DB = build_firestore_layer(
    RUN_COLLECTION, database=constants.RUN_DATABASE, project=constants.DEFAULT_PROJECT
)

GCS_STATS_DB = build_firestore_layer(
    GCS_STATS_COLLECTION, database=constants.RUN_DATABASE, project=constants.DEFAULT_PROJECT
)
