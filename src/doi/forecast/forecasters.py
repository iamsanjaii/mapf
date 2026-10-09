"""Forecasters: one question per case ("will the fleet's saving reach the price?"), several ways to answer it.

  numeric   the arithmetic extrapolation of the ledger (what rof_p uses)
  keyword   reads notices by matching cue phrases; falls back to numeric
  ledger    a threshold on the saving already recorded (one constant, fixed on the dev split)
  oracle    the truth label (the best a forecaster can be)
  inverted  the opposite of the truth (the worst)
"""
from typing import TYPE_CHECKING, Optional

from src.doi.forecast.agent import PROMPT_VERSION, run_agent
from src.doi.forecast.case import ForecastCase
from src.doi.forecast.result import ForecastResult
from src.doi.llm.chatcache import CachedChat
from src.doi.llm.client import Chat
from src.doi.notices import DROP_CUES, SURGE_CUES

if TYPE_CHECKING:
    from src.doi.config import SimConfig


def plain(answer: bool) -> ForecastResult:
    return ForecastResult(answer, None, "", "", 0.0, 0, 0, 0, 0, "")


def _truth(case: ForecastCase) -> bool:
    if case.truth is None:
        raise ValueError("this forecaster needs a case with a truth label")
    return case.truth


class NumericForecaster:
    name = "numeric"

    def forecast(self, case: ForecastCase) -> ForecastResult:
        return plain(case.numeric_forecast)


LEDGER_THRESHOLD = 2.0       # fixed on dev seeds 0..19 (88% there); never refit on test or human


class LedgerForecaster:
    """Yes when the recorded trips already show a saving above a fixed threshold. It reads no notice. It is the
    bar a model has to clear to claim it adds anything beyond the ledger."""
    name = "ledger"

    def forecast(self, case: ForecastCase) -> ForecastResult:
        return plain(case.ledger["saving_on_recorded_trips"] > LEDGER_THRESHOLD)


class OracleForecaster:
    name = "oracle"

    def forecast(self, case: ForecastCase) -> ForecastResult:
        return plain(_truth(case))


class InvertedForecaster:
    name = "inverted"

    def forecast(self, case: ForecastCase) -> ForecastResult:
        return plain(not _truth(case))


def group_word(case: ForecastCase) -> Optional[str]:
    """The word a notice would use for the obstacle's region: "west north bays" holds its row -> "north".

    A zone's group is its name without the first word; a one-word zone is its own group."""
    for name in sorted(case.zones):
        lo, hi = case.zones[name]["rows"]
        if lo <= case.obstacle[0] <= hi:
            words = name.split()
            return (words[1:] or words)[0].lower()
    return None


class KeywordForecaster:
    """The answer a team gets with no model: among notices that name the obstacle's region, a drop cue says no,
    else a surge cue says yes, else the numeric forecast stands."""
    name = "keyword"

    def forecast(self, case: ForecastCase) -> ForecastResult:
        word = group_word(case)
        if word is not None:
            texts = [text.lower() for _nid, _tick, text in case.notices if word in text.lower()]
            if any(cue in t for t in texts for cue in DROP_CUES):
                return plain(False)
            if any(cue in t for t in texts for cue in SURGE_CUES):
                return plain(True)
        return plain(case.numeric_forecast)


class LlmForecaster:
    """A language model behind the agent loop. `chat` is a CachedChat in a run, a fake in a test."""

    def __init__(self, model_key: str, chat: Chat, mode: str = "tools") -> None:
        self.name = f"llm:{model_key}"
        self.chat = chat
        self.mode = mode

    def forecast(self, case: ForecastCase) -> ForecastResult:
        return run_agent(case, self.chat, self.mode)


FORECASTERS = {"numeric": NumericForecaster, "keyword": KeywordForecaster, "ledger": LedgerForecaster,
               "oracle": OracleForecaster,
               "inverted": InvertedForecaster}


def make_forecaster(name: str, cfg: "Optional[SimConfig]" = None):
    """A forecaster by name. `llm:<key>` needs the run's config: it says where the store is, whether a miss may
    call the model, and the agent mode. Replay needs no credentials; only a live cache builds the real client."""
    if name.startswith("llm:"):
        if cfg is None:
            raise ValueError(f"the llm forecaster {name!r} needs the run's SimConfig (store, mode, live)")
        key = name.split(":", 1)[1]
        client = None
        if cfg.agent_live:
            from src.doi.llm.client import client_from_env
            client = client_from_env(key)
        return LlmForecaster(key, CachedChat(client, cfg.agent_cache, key, cfg.agent_live, PROMPT_VERSION),
                             cfg.agent_mode)
    if name not in FORECASTERS:
        raise ValueError(f"unknown forecaster {name!r}: choose from {', '.join(sorted(FORECASTERS))} or llm:<key>")
    return FORECASTERS[name]()
