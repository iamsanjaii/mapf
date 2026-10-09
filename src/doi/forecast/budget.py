"""A bound on what a script that can call a model would spend, and the question to ask before it does."""
from typing import Callable, Dict

EST_PROMPT_TOKENS, EST_COMPLETION_TOKENS = 1200, 80       # per call; a rough upper figure for the tools prompt


def call_budget(requests: int, calls_per_forecast: int) -> Dict[str, int]:
    calls = requests * calls_per_forecast
    return {"requests": requests, "calls": calls, "prompt_tokens": calls * EST_PROMPT_TOKENS,
            "completion_tokens": calls * EST_COMPLETION_TOKENS}


def confirm_calls(requests: int, calls_per_forecast: int, yes: bool = False, confirm: Callable[[str], str] = input,
                  say: Callable[[str], None] = print) -> bool:
    """Print the upper bound and ask. Nothing to spend asks nothing; --yes skips the question."""
    b = call_budget(requests, calls_per_forecast)
    say(f"{b['requests']} forecast requests, at most {b['calls']} model calls ({calls_per_forecast} per forecast); "
        f"about {b['prompt_tokens']} prompt and {b['completion_tokens']} completion tokens at most "
        f"(multiply by your model's price)")
    if b["calls"] == 0 or yes:
        return True
    return confirm("Proceed with these calls? [y/N] ").strip().lower() == "y"
