"""Some constants."""
import os

from zetta_utils import constants

PROJECT = constants.DEFAULT_PROJECT
DATABASE = os.environ.get("ANNOTATIONS_DB_NAME", "annotations-fs")
