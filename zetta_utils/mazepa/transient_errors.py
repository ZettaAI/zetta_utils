from typing import Final, Type

import attrs

MAX_TRANSIENT_RETRIES: Final = 80


@attrs.mutable
class TransientErrorCondition:
    exception_type: Type[BaseException]
    text_signature: str = ""

    def does_match(self, exc: BaseException):
        cur: BaseException | None = exc
        while cur is not None:
            if isinstance(cur, self.exception_type) and self.text_signature in str(cur):
                return True
            cur = cur.__cause__ or cur.__context__
        return False


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
        # CUDA context lost (e.g. SLURM preemption changed CUDA_VISIBLE_DEVICES
        # mid-process). PyTorch surfaces this as "CUDA unknown error".
        exception_type=RuntimeError,
        text_signature="CUDA unknown error",
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
        # botocore ReadTimeoutError — same root cause as urllib3 above but
        # different wording: "Read timeout on endpoint URL: ...".
        exception_type=Exception,
        text_signature="Read timeout on endpoint URL",
    ),
    TransientErrorCondition(
        # urllib3 protocol error from a partial response.
        exception_type=Exception,
        text_signature="ProtocolError",
    ),
    TransientErrorCondition(
        # Server refused a new connection (nokura/c10s under load).
        exception_type=Exception,
        text_signature="Connection refused",
    ),
    TransientErrorCondition(
        # boto3 / botocore connection failure during upload/delete.
        # Surfaces from `botocore.exceptions.EndpointConnectionError`; its
        # str() is "Could not connect to the endpoint URL: ..."
        exception_type=Exception,
        text_signature="Could not connect to the endpoint URL",
    ),
    TransientErrorCondition(
        # botocore.exceptions.ConnectionClosedError — server dropped the
        # TCP connection mid-request.
        exception_type=Exception,
        text_signature="Connection was closed before we received a valid response",
    ),
    TransientErrorCondition(
        # Generic socket / DNS resolution glitches.
        exception_type=Exception,
        text_signature="Temporary failure in name resolution",
    ),
    TransientErrorCondition(
        # Transient SSL handshake failure (nokura/c10s under load).
        exception_type=Exception,
        text_signature="SSL validation failed",
    ),
    TransientErrorCondition(
        # ssl.SSLEOFError — peer closed TLS connection mid-handshake/transfer.
        # Surfaces as "EOF occurred in violation of protocol (_ssl.c:NNNN)".
        exception_type=Exception,
        text_signature="EOF occurred in violation of protocol",
    ),
    TransientErrorCondition(
        # CV/cloudfiles wraps SSL/network errors as "SSL validation failed for ..."
        # — covers transient TLS faults from nokura/c10s.
        exception_type=Exception,
        text_signature="SSL validation failed for",
    ),
    TransientErrorCondition(
        # S3 upload checksum mismatch — data corrupted in transit.
        exception_type=Exception,
        text_signature="XAmzContentSHA256Mismatch",
    ),
)
