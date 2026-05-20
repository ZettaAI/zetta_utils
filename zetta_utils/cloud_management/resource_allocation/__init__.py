"""Resource allocation subpackage — lazily resolved via zetta_utils.common.lazy."""
from typing import TYPE_CHECKING

from zetta_utils.common.lazy import make_lazy_module

_LAZY_SUBPACKAGES = ("aws_sqs", "gcloud", "k8s")

__getattr__, __dir__ = make_lazy_module(__name__, globals(), _LAZY_SUBPACKAGES)

if TYPE_CHECKING:
    from . import aws_sqs, gcloud, k8s
