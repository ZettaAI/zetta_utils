from typing import Final, Type

import attrs

MAX_TRANSIENT_RETRIES: Final = 40


@attrs.mutable
class TransientErrorCondition:
    exception_type: Type[BaseException]
    text_signature: str = ""

    def does_match(self, exc: BaseException):
        return isinstance(exc, self.exception_type) and self.text_signature in str(exc)


class ExplicitTransientError(Exception):
    ...


TRANSIENT_ERROR_CONDITIONS: Final = (
    TransientErrorCondition(
        # If running on GPU spot instance: Graceful shutdown failed
        exception_type=ExplicitTransientError,
    ),
    TransientErrorCondition(
        # If running on GPU spot instance: Graceful shutdown failed
        exception_type=RuntimeError,
        text_signature="Found no NVIDIA driver on your system",
    ),
    TransientErrorCondition(
        # If running on GPU spot instance: Graceful shutdown failed
        exception_type=RuntimeError,
        text_signature="Attempting to deserialize object on a CUDA device",
    ),
    TransientErrorCondition(
        # Transient GCS error
        exception_type=Exception,
        text_signature="You have exceeded your bucket's allowed rate",
    ),
    TransientErrorCondition(
        # Transient GCS error
        exception_type=Exception,
        text_signature="We encountered an internal error. Please try again",
    ),
    TransientErrorCondition(
        # Transient GCS error
        exception_type=Exception,
        text_signature="Compute Engine Metadata server unavailable",
    ),
    TransientErrorCondition(
        exception_type=OSError,
        text_signature="Input/output error",
    ),
)
