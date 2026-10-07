# Forecast datasets

| file | origin | use |
|---|---|---|
| `human_notices.jsonl` | written by people | notice texts for the held-out `human` cases |

`dev.jsonl`, `test.jsonl` and `human.jsonl` (forecast cases) are produced by `experiments/doi_agent_cases.py`. They are generated, not committed (about 10 MB for `test`); each line is `{"meta": {split, seed, mode, bank, stalled}, "case": {...}}`. `dev` (seeds 0..19) is for prompt development only; `test` (seeds 100..149) and `human` (seeds 300..319) are held-out. Seeds 200..229 belong to E9. Score them with `experiments/doi_agent_eval.py`. Without `human_notices.jsonl` the `human` split is skipped and level 2 is reported as not done.

## Protocol for `human_notices.jsonl`

No code generates this file. It is collected by hand and committed when ready.

* One JSON object per line: `{"kind": "surge" | "drop" | "distractor", "group": "north bays" | "south bays" | "", "text": "..."}`.
* At least 100 texts: 40 surge, 40 drop, 20 distractor.
* Writers have not seen the templates in `src/doi/notices.py`.
* Each writer is told the kind and the zone group and writes one short message as they would on a radio or chat.
  A surge says work is about to move into that group. A drop says work there is about to stop. A distractor is
  any message that does not change where robots will go; its `group` is empty.
* No real names, sites or personal data. The texts are sent to a hosted API.

## Stand-in: notices written by language models

`experiments/doi_make_notices.py` asks language models (the keys `small` and `large`, in turn) for the same kinds of
messages and writes `llm_notices.jsonl`, with the model that wrote each row in `source`. It is a stand-in when no
people are available. **These texts are not human-written**: do not report results on them as the `human` set or as
level 2 of the design. To run the case collector on them, pass `--human-notices data/forecasts/llm_notices.jsonl`,
and say in the write-up that a model was scored on text a model wrote.
