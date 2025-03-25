import signal
from contextlib import contextmanager


@contextmanager
def custom_signal_handler_ctx(fn, target_signal):  # pragma: no cover
    original_sigint_handler = signal.getsignal(target_signal)  # eg signal.SIGINT

    signal.signal(target_signal, fn)

    try:
        yield
    except Exception as e:
        raise e
    finally:
        signal.signal(target_signal, original_sigint_handler)
