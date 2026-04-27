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
    pass


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
        # I/O error, often from GKE Image Streaming
        exception_type=OSError,
        text_signature="Input/output error",
    ),
    TransientErrorCondition(
        # GCE metadata service unavailable
        exception_type=Exception,
        text_signature="Failed to retrieve",
    ),
    TransientErrorCondition(
        # Transient CUDA errors
        exception_type=RuntimeError,
        text_signature="CUDA error:",
    ),
    TransientErrorCondition(
        # GCS request hit the mproxy sidecar during a restart window — the
        # proxy is coming back up (either via self-heal or K8s OnFailure)
        # within seconds. Retrying via SQS visibility timeout lands on a
        # live proxy, so this is safe to treat as transient.
        exception_type=Exception,
        text_signature="Unable to connect to proxy",
    ),
    TransientErrorCondition(
        # mproxy CA bundle not yet visible to the worker — wait_for_ca
        # fired but the combined CA file lookup races the OAuth refresh.
        # Retrying after the visibility timeout lands once the bundle is
        # in place.
        exception_type=Exception,
        text_signature="CERTIFICATE_VERIFY_FAILED",
    ),
    TransientErrorCondition(
        # OAuth/metadata server transiently dropped the connection during
        # token refresh. Surfaces from `google.auth.exceptions.TransportError`
        # via gcsfs/google-auth.
        exception_type=Exception,
        text_signature="Remote end closed connection without response",
    ),
    TransientErrorCondition(
        # urllib3 transient network errors (typically nokura/c10s server
        # dropping a TCP connection mid-transfer).
        exception_type=Exception,
        text_signature="Connection aborted",
    ),
    TransientErrorCondition(
        # urllib3 read timeout — server took too long to respond. Retrying
        # the SQS visibility timeout typically lands on a healthy connection.
        exception_type=Exception,
        text_signature="Read timed out",
    ),
    TransientErrorCondition(
        # urllib3 protocol error from a partial response.
        exception_type=Exception,
        text_signature="ProtocolError",
    ),
)
