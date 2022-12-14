import signal
from contextlib import contextmanager

from .user_input import get_user_input


# context manager for capturing the SIGINT signal (thrown by ctrl+c)
@contextmanager
def confirm_sigint_ctx():  # pragma: no cover
    # save original SIGINT handler
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    # define new signal handler
    def handle_sigint(_, __):
        interrupt = False
        try:
            user_input = get_user_input(
                prompt="Confirm sending KeyboardInterrupt (y/[n])? ", timeout=7
            )
            if user_input is None:
                print("\nNo input for 7 seconds. Resuming...")
            elif user_input != "y":
                print("Resuming...")
            else:
                interrupt = True
        except KeyboardInterrupt:
            interrupt = True
        if interrupt:
            raise KeyboardInterrupt

    # install the new SIGINT handler
    signal.signal(signal.SIGINT, handle_sigint)

    try:
        # whatever is yielded is what's picked up in the "as"
        yield
    except Exception as e:
        raise e
    finally:
        # when leaving contextmanager, reinstall old SIGINT handler
        # note: only works if old SIGINT handler was originally installed by Python
        signal.signal(signal.SIGINT, original_sigint_handler)
