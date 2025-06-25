# Import from completion module
from .completion import (
    register_completion_handler,
    handle_task_completion,
    get_completion_handler,
    CompletionResult
)

# Import from verification module
from .verification import (
    register_verifier,
    verify_task,
    get_verifier,
    VerificationResult
)

# Import task-specific handlers to register them
from . import seg_trace_v0

__all__ = [
    # Completion exports
    "register_completion_handler",
    "handle_task_completion",
    "get_completion_handler",
    "CompletionResult",
    # Verification exports
    "register_verifier",
    "verify_task",
    "get_verifier",
    "VerificationResult"
]