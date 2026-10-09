import dataclasses
import pytest
from src.doi.forecast.case import ForecastCase
from src.doi.forecast.forecasters import (InvertedForecaster, KeywordForecaster, NumericForecaster, OracleForecaster,
                                          group_word, make_forecaster)

ZONES = {"west north bays": {"rows": (0, 6), "cols": (0, 9)}, "west south bays": {"rows": (8, 14), "cols": (0, 9)},
         "east north bays": {"rows": (0, 6), "cols": (11, 20)}, "east south bays": {"rows": (8, 14), "cols": (11, 20)}}
BASE = ForecastCase(case_id="x", robot=0, tick=1, mode="push", kind="pallet", obstacle=(10, 10), landing=(10, 11),
                    plan_key=(((10, 10), (0, 1), 1),), price=10.0, known_saving=6.0, notices=(), ledger={},
                    zones=ZONES, trip_saving={}, numeric_forecast=False, truth=True)


def case(**kw):
    return dataclasses.replace(BASE, **kw)


def test_plain_forecasters():
    assert NumericForecaster().forecast(case(numeric_forecast=True)).answer is True
    assert NumericForecaster().forecast(case(numeric_forecast=False)).answer is False
    assert OracleForecaster().forecast(case(truth=True)).answer is True
    assert InvertedForecaster().forecast(case(truth=True)).answer is False
    res = OracleForecaster().forecast(case())
    assert (res.failed, res.latency_s, res.calls, res.confidence) == ("", 0.0, 0, None)
    for cls in (OracleForecaster, InvertedForecaster):
        with pytest.raises(ValueError):
            cls().forecast(case(truth=None))


def test_group_word_comes_from_the_zone_that_holds_the_obstacle_row():
    assert group_word(case(obstacle=(10, 10))) == "south" and group_word(case(obstacle=(2, 10))) == "north"
    assert group_word(case(obstacle=(7, 10))) is None and group_word(case(zones={})) is None


def test_keyword_reads_only_notices_about_its_own_group():
    kw = KeywordForecaster()
    surge = ("n0", 5, "FYI: wave 2 is all in the south bays")
    drop = ("n1", 5, "the north bays are finished after this wave")
    assert kw.forecast(case(notices=(surge, drop))).answer is True              # south obstacle, surge for south
    assert kw.forecast(case(obstacle=(2, 10), notices=(surge, drop), numeric_forecast=True)).answer is False
    assert kw.forecast(case(notices=(drop,), numeric_forecast=False)).answer is False     # nothing about south
    assert kw.forecast(case(notices=(drop,), numeric_forecast=True)).answer is True       # falls back to numeric


def test_keyword_puts_a_drop_before_a_surge_and_ignores_unknown_wording():
    kw = KeywordForecaster()
    both = (("a", 1, "the south bays get busy after the break"), ("b", 2, "NO MORE orders for the south bays"))
    assert kw.forecast(case(notices=both, numeric_forecast=True)).answer is False
    reworded = (("a", 1, "afternoon orders are concentrated on the south bays"),)
    assert kw.forecast(case(notices=reworded, numeric_forecast=False)).answer is False    # no dev cue: numeric
    assert kw.forecast(case(zones={}, notices=both, numeric_forecast=True)).answer is True  # no zones: numeric


def test_make_forecaster_names():
    for name in ("numeric", "keyword", "ledger", "oracle", "inverted"):
        assert make_forecaster(name).name == name
    with pytest.raises(ValueError, match="llm"):
        make_forecaster("llm:small")
    with pytest.raises(ValueError):
        make_forecaster("crystal_ball")


def test_ledger_forecaster_answers_from_the_recorded_saving_alone():
    import dataclasses
    from src.doi.forecast.forecasters import LEDGER_THRESHOLD, make_forecaster
    f = make_forecaster("ledger")
    low = dataclasses.replace(BASE, ledger={**BASE.ledger, "saving_on_recorded_trips": LEDGER_THRESHOLD})
    high = dataclasses.replace(BASE, ledger={**BASE.ledger, "saving_on_recorded_trips": LEDGER_THRESHOLD + 0.5},
                               notices=(("n0", 1, "radio: picking is done in the north bays"),))
    assert f.forecast(low).answer is False and f.forecast(high).answer is True
