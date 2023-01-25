import signal
from contextlib import contextmanager


@contextmanager
def custom_signal_handler_ctx(fn, target_signal):  # pragma: no cover
    # save original SIGINT handler
    original_sigint_handler = signal.getsignal(target_signal)  # eg signal.SIGINT

    # install the new SIGINT handler
    signal.signal(signal.SIGINT, fn)

    try:
        # whatever is yielded is what's picked up in the "as"
        yield
    except Exception as e:
        raise e
    finally:
        # when leaving contextmanager, reinstall old SIGINT handler
        # note: only works if old SIGINT handler was originally installed by Python
        signal.signal(signal.SIGINT, original_sigint_handler)
