from __future__ import annotations

import threading
import time

import attrs


@attrs.mutable
class Timer:  # pragma: no cover
    start: float = 0.0
    elapsed: float = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


class RepeatTimer(threading.Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)
