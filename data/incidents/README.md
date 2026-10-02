# Incident report datasets

| file | origin | seeds | use |
|---|---|---|---|
| `dev.jsonl` | templates (`src/doi/incidents.py`) | 0..19, `incidents_aisles` and `incidents_room`, `p_false = 0.2` | prompt development only |
| `test.jsonl` | templates | 100..149, same settings | held-out scoring, first scored after the `prereg-v1` tag |
| `human.jsonl` | written by people | n/a | held-out scoring, always reported separately from `test` |

Seeds 200..229 are reserved for the simulation experiments E7 and E8 and never appear in dev or test.
Regenerate dev and test with `python experiments/doi_build_incidents.py` (deterministic).

Known template caveat: in `incidents_aisles` the kind `rack_damage` occurs exactly when the class is
`needs_human`, so a model can recover the class from the kind phrase alone. The class cue list in
`CLASS_CUES` is added on top. Report this when quoting class accuracy on `test`; `human.jsonl` is the
check against it.

## Protocol for `human.jsonl`

No code generates this file. It is collected by hand and committed when ready.

* No real names, real sites or personal data. Items are sent to a hosted API.
* At least 100 items, written by people who have not seen the templates.
* Each writer gets a map image, a location name, a kind and a class, and writes one short message as
  they would on a radio or chat.
* About 10% of writers are told to leave the bay out (ambiguous report; the expected answer is a reject).
* The file uses the `IncidentItem` schema (see `src/doi/incidents.py`) with `split = "human"` and
  `source = "human"`.
