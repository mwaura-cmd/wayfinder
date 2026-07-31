from datetime import datetime, timezone
from typing import Optional

class IterationGovernor:
    def __init__(self, max_steps: int, timeout_seconds: int):
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.steps = 0
        self.malformed_count = 0
        self.start_time = datetime.now(timezone.utc)

    def tick(self) -> None:
        self.steps += 1

    def record_malformed(self) -> None:
        self.malformed_count += 1

    def is_timed_out(self) -> bool:
        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return elapsed > self.timeout_seconds

    def reset_malformed(self) -> None:
        self.malformed_count = 0

    def status_summary(self) -> str:
        if self.is_timed_out():
            return f"Timeout reached after {self.timeout_seconds}s"
        if self.steps >= self.max_steps:
            return f"Max iterations reached at step {self.max_steps}"
        if self.malformed_count >= 3:
            return "Loop aborted — 3 consecutive parse failures"
        return f"Running... Step {self.steps}"
