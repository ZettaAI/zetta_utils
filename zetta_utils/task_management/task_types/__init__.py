# Import from completion module
from .completion import (
    register_completion_handler,
    handle_task_completion,
    get_completion_handler,
)

# Import from verification module
from .verification import register_verifier, verify_task, get_verifier, VerificationResult

# Import from creation module
from .creation import (
    register_creation_handler,
    add_segment_task,
    get_creation_handler,
    CreationResult,
)

# Import task-specific handlers to register them
from . import trace_v0
from . import trace_feedback_v0

__all__ = [
    # Completion exports
    "register_completion_handler",
    "handle_task_completion",
    "get_completion_handler",
    # Verification exports
    "register_verifier",
    "verify_task",
    "get_verifier",
    "VerificationResult",
    # Creation exports
    "register_creation_handler",
    "add_segment_task",
    "get_creation_handler",
    "CreationResult",
]
