from __future__ import annotations

import contextlib
import os
import signal
import sys
from typing import Generator, Protocol

import rich
from rich import progress

from zetta_utils.common import custom_signal_handler_ctx, get_user_confirmation

from .execution_state import ProgressReport


def get_confirm_sigint_fn(progress_bar: progress.Progress):  # pragma: no cover
    def handler(_, __):
        progress_bar.stop()
        interrupt = False
        try:
            confirmed = get_user_confirmation(
                prompt="Confirm sending KeyboardInterrupt? ", timeout=7
            )
            if confirmed is None:
                rich.print("\nNo input for 7 seconds. Resuming...")
            elif not confirmed:
                rich.print("Resuming...")
            else:
                interrupt = True
        except KeyboardInterrupt:
            interrupt = True

        if interrupt:
            raise KeyboardInterrupt
        progress_bar.start()

    return handler


class ProgressUpdateFN(Protocol):
    def __call__(self, progress_reports: dict[str, ProgressReport]) -> None:
        ...


@contextlib.contextmanager
def progress_ctx_mngr(
    expected_total_counts: dict[str, int]
) -> Generator[ProgressUpdateFN, None, None]:  # pragma: no cover
    progress_bar = progress.Progress(
        progress.SpinnerColumn(),
        progress.TextColumn("[progress.description]{task.description}"),
        progress.BarColumn(),
        progress.TaskProgressColumn(),
        progress.TextColumn("[progress.description] {task.completed}/{task.total}"),
        progress.TimeElapsedColumn(),
        progress.TimeRemainingColumn(),
        transient=True,
        # refresh_per_second=1,
    )

    def custom_debugger_hook():
        # reading the value of the environment variable
        val = os.environ.get("PYTHONBREAKPOINT")
        # if the value has been set to 0, skip all breakpoints
        if val == "0":
            return None
        # else if the value is an empty string, invoke the default pdb debugger
        elif val is not None and val != "pdb.set_trace":
            raise Exception("Custom debuggers are not allowed when `rich.progress is in use.`")

        progress_bar.stop()
        import pdb  # pylint: disable=import-outside-toplevel

        return pdb.Pdb().set_trace(sys._getframe().f_back)  # pylint: disable=protected-access

    sys.breakpointhook = custom_debugger_hook

    with progress_bar as progress_bar:
        with custom_signal_handler_ctx(get_confirm_sigint_fn(progress_bar), signal.SIGINT):
            submission_tracker_ids = {
                k: progress_bar.add_task(
                    f"[cyan]Submission {k}",
                    total=v,
                    start=False,
                    spinner_color="cyan",
                    auto_refresh=False,
                )
                for k, v in expected_total_counts.items()
            }

            execution_tracker_ids: dict[str, progress.TaskID] = {}

            def update_fn(progress_reports: dict[str, ProgressReport]) -> None:
                if not hasattr(sys, "gettrace") or sys.gettrace() is None:
                    progress_bar.start()

                for k, v in progress_reports.items():
                    if k in submission_tracker_ids:
                        progress_bar.start_task(submission_tracker_ids[k])
                        progress_bar.update(submission_tracker_ids[k], completed=v.submitted_count)

                    if k not in execution_tracker_ids:
                        execution_tracker_ids[k] = progress_bar.add_task(
                            f"[green]Completion {k}", start=False, spinner_color="green"
                        )

                    progress_bar.update(execution_tracker_ids[k], total=v.submitted_count)

                    if v.completed_count != 0:
                        progress_bar.start_task(execution_tracker_ids[k])
                        progress_bar.update(execution_tracker_ids[k], completed=v.completed_count)

                progress_bar.refresh()

            yield update_fn
            try:
                progress_bar.stop()
            except IndexError:
                pass
