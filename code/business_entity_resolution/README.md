# business_entity_resolution (F05 implementation)

Shared production package. Notebooks become launch/report interfaces only.

Layout per `F05_098_IMPLEMENTATION_PLAN.md` 16.1. Key contracts (16.2):

- `normalize(record, policy)` -> `normalization.normalize_record`
- `retrieve(query_batch, index_bundle, config)` -> `retrieval.lexical.retrieve_channel`
- `build_features` -> `features.pair_feature_row`
- `score` -> `training.matcher`
- `decide` -> `calibration.decide_two_threshold`
- `evaluate` -> `evaluation.summarize` + `metrics.macro_f05`
- `write_outputs` -> `io.write_id_lists`

Pinned CPU runtime: see `requirements.txt` (validated local env 2026-09-26).
GPU/transformer training lives in its own validated env (not this pin set).
