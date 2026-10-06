"""Forecasters: one question per case ("will the fleet's saving reach the price?"), several ways to answer it.

  numeric   the arithmetic extrapolation of the ledger (what rof_p uses)
  keyword   reads notices by matching cue phrases; falls back to numeric
  oracle    the truth label (the best a forecaster can be)
  inverted  the opposite of the truth (the worst)
"""
from typing import Optional

from src.doi.forecast.case import ForecastCase
from src.doi.forecast.result import ForecastResult
from src.doi.notices import DROP_CUES, SURGE_CUES


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


FORECASTERS = {"numeric": NumericForecaster, "keyword": KeywordForecaster, "oracle": OracleForecaster,
               "inverted": InvertedForecaster}


def make_forecaster(name: str):
    if name.startswith("llm:"):
        raise ValueError("llm forecasters are not built yet (Part 2 of the forecast-guard plan)")
    if name not in FORECASTERS:
        raise ValueError(f"unknown forecaster {name!r}: choose from {', '.join(sorted(FORECASTERS))}")
    return FORECASTERS[name]()
