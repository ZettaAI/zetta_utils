"""Message queues subpackage exports — lazily resolved via zetta_utils.common.lazy."""
from typing import TYPE_CHECKING

from zetta_utils.common.lazy import make_lazy_module

_LAZY_SUBPACKAGES = ("serialization",)

_LAZY_REEXPORTS = {
    ".base": ("ReceivedMessage", "MessageQueue", "TQTask"),
    ".file": ("FileQueue",),
    ".sqs": ("SQSQueue",),
}

__getattr__, __dir__ = make_lazy_module(
    __name__, globals(), _LAZY_SUBPACKAGES, _LAZY_REEXPORTS
)

if TYPE_CHECKING:
    from . import serialization
    from .base import MessageQueue, ReceivedMessage, TQTask
    from .file import FileQueue
    from .sqs import SQSQueue
