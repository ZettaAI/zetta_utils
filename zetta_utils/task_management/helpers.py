from google.api_core import exceptions
from google.cloud.firestore import Transaction
from tenacity import (
    retry,
    retry_any,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from .project import get_firestore_client


def get_transaction() -> Transaction:
    """Get a transaction from the Firestore client.

    :return: A new transaction object
    """
    client = get_firestore_client()
    return client.transaction()


# Retry decorator for Firestore operations
retry_transient_errors = retry(
    retry=retry_any(
        retry_if_exception_type(
            (exceptions.Aborted, exceptions.ServiceUnavailable, exceptions.DeadlineExceeded)
        ),
        retry_if_exception(
            lambda x: isinstance(x, ValueError)
            and "Failed to commit transaction in 5 attempts." in str(x)
        ),
    ),
    wait=wait_exponential(multiplier=0.1, min=0.2, max=5, exp_base=3) + wait_random(0, 0.4),
    stop=stop_after_attempt(8),
)
