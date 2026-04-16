import signal
from contextlib import contextmanager


def configure_pool_signals():  # pragma: no cover
    """Initializer for `multiprocessing.Pool` children. SIGINT is ignored
    so terminal Ctrl-C doesn't kill them — the head process owns the
    confirm decision. SIGTERM/SIGHUP stay at SIG_DFL so `pool.terminate()`
    and explicit `kill` work as expected.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGHUP, signal.SIG_DFL)


@contextmanager
def custom_signal_handler_ctx(fn, target_signal):  # pragma: no cover
    original_signal_handler = signal.getsignal(target_signal)  # eg signal.SIGINT

    signal.signal(target_signal, fn)

    try:
        yield
    except Exception as e:
        raise e
    finally:
        signal.signal(target_signal, original_signal_handler)
