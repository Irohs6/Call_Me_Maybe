"""Profile Call Me Maybe without changing its execution logic.

Run from the repository root: uv run python -m scripts.profile
Existing CLI flags can be passed through unchanged.
"""

from dataclasses import dataclass
from functools import wraps
from time import perf_counter
from typing import Any, Callable
from unittest.mock import patch


@dataclass
class Measurement:
    calls: int = 0
    seconds: float = 0.0
    slowest: float = 0.0


def timed(
    method: Callable[..., Any], measurement: Measurement,
) -> Callable[..., Any]:
    """Wrap a method with cumulative wall-clock measurements."""

    @wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = perf_counter()
        try:
            return method(*args, **kwargs)
        finally:
            elapsed = perf_counter() - started
            measurement.calls += 1
            measurement.seconds += elapsed
            measurement.slowest = max(measurement.slowest, elapsed)

    return wrapper


def report(measurements: dict[str, Measurement], total: float) -> None:
    """Print cumulative times and call counts for the selected methods."""
    print("\n=== Profiling (wall-clock time) ===")
    print(f"Total elapsed: {total:.2f} s")
    print(
        f"{'Method':<31} {'Calls':>9} {'Total (s)':>12} "
        f"{'Avg (ms)':>12} {'Share':>9}"
    )
    for name, measurement in measurements.items():
        average = (
            1000 * measurement.seconds / measurement.calls
            if measurement.calls else 0.0
        )
        share = 100 * measurement.seconds / total if total else 0.0
        print(
            f"{name:<31} {measurement.calls:>9} "
            f"{measurement.seconds:>12.2f} {average:>12.2f} "
            f"{share:>8.1f}%"
        )
    other = total - sum(item.seconds for item in measurements.values())
    print(f"Other work / initialization: {max(0.0, other):.2f} s")
    print("A small profiling overhead is included; model output is unchanged.")


def main() -> int:
    """Run the normal CLI while timing only selected methods."""
    started = perf_counter()
    from llm_sdk import Small_LLM_Model
    from src.__main__ import main as run_application
    from src.json_checker import JsonChecker

    measurements = {
        "get_logits_from_input_ids": Measurement(),
        "decode": Measurement(),
        "JsonChecker.status": Measurement(),
    }

    with (
        patch.object(
            Small_LLM_Model,
            "get_logits_from_input_ids",
            timed(
                Small_LLM_Model.get_logits_from_input_ids,
                measurements["get_logits_from_input_ids"],
            ),
        ),
        patch.object(
            Small_LLM_Model,
            "decode",
            timed(Small_LLM_Model.decode, measurements["decode"]),
        ),
        patch.object(
            JsonChecker,
            "status",
            timed(JsonChecker.status, measurements["JsonChecker.status"]),
        ),
    ):
        try:
            return run_application()
        finally:
            report(measurements, perf_counter() - started)


if __name__ == "__main__":
    raise SystemExit(main())
