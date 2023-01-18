from __future__ import annotations
import signal
import contextlib
from typing import Protocol, Generator
import rich
from rich import progress

from .execution_state import ProgressReport

from zetta_utils.common import get_user_confirmation
from zetta_utils.common import custom_signal_handler_ctx


def get_confirm_sigint_fn(progress_bar: progress.Progress):
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
def progress_ctx(expected_total_counts: dict[str, int]) -> Generator[ProgressUpdateFN, None, None]:
    progress_ctx_mngr = progress.Progress(
        progress.SpinnerColumn(),
        progress.TextColumn("[progress.description]{task.description}"),
        progress.BarColumn(),
        progress.TaskProgressColumn(),
        progress.TextColumn("[progress.description] {task.completed}/{task.total}"),
        progress.TimeElapsedColumn(),
        progress.TimeRemainingColumn(),
        transient=True,
    )

    with progress_ctx_mngr as progress_bar:
        with custom_signal_handler_ctx(get_confirm_sigint_fn(progress_bar), signal.SIGINT):
            submission_tracker_ids = {
                k: progress_bar.add_task(
                    f"[cyan]Submission {k}",
                    total=v,
                    start=False,
                    spinner_color="cyan",
                    auto_refresh=False
                )
                for k, v in expected_total_counts.items()
            }

            execution_tracker_ids: dict[str, progress.TaskID] = {}

            def update_fn(progress_reports: dict[str, ProgressReport]) -> None:
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

            try:
                yield update_fn
            finally:
                progress_bar.stop()
