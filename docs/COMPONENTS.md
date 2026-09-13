# Components and reading boundaries

This is a static source map, not a setup guide. Dependencies were not installed and application code was not imported or executed. Python parsing checks syntax only. JavaScript/JSX was read as text, without a browser or build. CI is not configured by design; release checks are local.

## Collection and representations

| Source | Behavior represented in code |
| --- | --- |
| [config.py](../config.py) | Relative data paths, service endpoints, curated seeds/tag groups, thresholds, 1024-dimensional MERT configuration |
| [src/db.py](../src/db.py) | SQLite schema, collection progress, track/user joins, Deezer and scrobble migrations |
| [src/lastfm_api.py](../src/lastfm_api.py) | Rate-limited API requests for users, friends, top tracks, artist metadata, and timestamped recent tracks |
| [src/scraper.py](../src/scraper.py) | Historical listener-page scraping; inclusion does not establish permission to use it |
| [src/deezer_api.py](../src/deezer_api.py) | Search, fresh preview URL lookup, preview download, retry/backoff logic |
| [src/embedder.py](../src/embedder.py) | External MERT loading, audio decoding, last-state time pooling, L2 normalization |
| [scripts](../scripts) | Discovery, aggregate counts, diversity, matching/embedding, overlap snowball, conversion, and progress reports |
| [sequential/scripts](../sequential/scripts) | Event collection and progress only |

Top-track collection produces aggregate all-time counts, not a chronology. Scrobble records carry Unix timestamps, with deduplication by `(user_id, lastfm_id, listened_at)`. A track key is constructed as `artist::title`; names and string matching are imperfect identifiers. Collection uses human-chosen seeds and filters. Preview coverage is a selection mechanism, not a complete music catalog.

Rate limits and runtime estimates in source are historical assumptions, not current service guarantees. Some modules create log directories and file handlers at import time. A `--dry-run` name does not guarantee no I/O: the snowball path resolves seeds using service calls and initializes storage before returning. Read [Data and rights](DATA_AND_RIGHTS.md) before considering any collection.

## Training and expected artifacts

The [library source](../library/src) implements score transforms, prepared per-user libraries, random windows, batching, the attention model, and two output-head choices. [Training scripts](../library/scripts) provide training loops and historical experiment orchestration. None of the following artifacts is supplied:

| Expected path | Schema or shape in retained code |
| --- | --- |
| `data/melody_matcher.db` | `users` (username, status, activity/diversity and cohort fields), `tracks` (track key, artist/title, Deezer match fields), `user_tracks` (user, track, count), artist tags and progress tables |
| `data/embeddings.h5` | `track_ids`: length `C` text keys; `embeddings`: float32 matrix `(C, 1024)` under the selected configuration |
| `data/embeddings.npy` | Optional float32 matrix aligned exactly with HDF5 rows; training prefers this when present |
| `data/training/embedding_index.json` | Track-key-to-row-index mapping |
| `data/training/user_libraries.h5` | Groups `user_<id>`, each with aligned `track_indices` (int32) and `scores` (float32) arrays |
| `data/training/user_splits.json` | `train`, `val`, `test` lists of numeric user IDs |
| `data/training/dataset_config.json` | Serialized dataset variant; variants also use `data/training/exp_<name>/` |
| `data/best_model.pt`, `data/sweeps/<sweep>/<trial>/best.pt` | Dictionaries containing `model_state_dict` and `model_config`; the latter selects architecture and model type |
| `data/checkpoints/checkpoint_epoch_*.pt` | Training checkpoints including optimizer/scheduler state and progress |
| Sweep `summary.json`, `metrics.jsonl`, `results.jsonl`, `results.csv` | Trial configuration/status, metrics, and results; absent historical evidence |

`C` is the available catalog size, not a fixed release constant. References to 2.2M tracks in source/UI are historical assumptions, not a bundled catalog. `library/data` and `sequential/data` are empty historical directory markers; the main code uses root `data`.

Batches use context embeddings `(B, N, 1024)`, context scores `(B, N, 1)`, candidate embeddings `(B, M, 1024)`, and Boolean padding masks `(B, N)` / `(B, M)`. `True` means padding. The hurdle target also carries `target_is_positive`; negatives and padding are false, with padding separately masked. Scoring outputs `(B, M, 1)`; the hurdle model returns `play_logits` and `score_pred` of that shape. See [Research](RESEARCH.md) for their actual target meaning.

## Experimental harness and coverage analysis

[Backend modules](../harness/backend) parse XML, fuzzy-match to embedding keys, discover local checkpoint files, score candidates, and persist runs and feedback. Matching strips variants and accepts a title WRatio threshold of 85, with an artist fallback threshold of 90. These heuristics can conflate recordings. The stored `match_policy` does not change the matching call in `app.py`.

The harness adds `harness_libraries`, `harness_library_tracks`, `harness_runs`, and `harness_feedback` to the shared database. It saves raw uploads in `data/harness_uploads/` and a pickle index in `data/harness_match_index.pkl`. Model records and errors can expose local paths. Importing the backend creates an upload directory; startup initializes storage and deserializes checkpoints. It is not a read-only viewer or a hardened service: wildcard CORS, no authentication, and no upload-size boundary are present.

[Frontend source](../harness/frontend) uses external React 18.3.1 and ReactDOM UMD scripts, Babel standalone 7.29.0, and Google Fonts, without an npm build. Opening it can make network requests. It sends uploaded XML to the configured backend, stores selection state in `localStorage`, and can prefetch inference runs during comparison. The “Play” control opens an Apple Music search link; it is not an implemented in-app playback system. “Production” badges, latency estimates, and catalog counts are historical UI labels, not release claims.

[Coverage comparison](../library_analysis/compare.py) expects a separately supplied `library_analysis/library.xml` and a read-only database connection. The input name was generalized; no example library is bundled. Its “embedded” result category is inferred from Deezer matching status, not verified against the embedding matrix, so it is not an embedding-coverage guarantee.

## Preserved defects and hazards

These are static findings, not a complete bug inventory or runtime security assessment:

- `hyperparam_sweep.py` unpacks two return values from `train_epoch` and `validate`, while current `train.py` returns three. This preexisting mismatch can fail a trial; syntax parsing cannot catch it.
- `prepare_dataset_variant` overwrites the saved dataset configuration before deciding to reuse existing HDF5/split files. File existence does not prove a complete or matching dataset. An interrupted preparation can leave partial artifacts.
- The diversity SQL bypass applies to any user with `diversity_score IS NULL`, not just a specifically identified snowball cohort. Split assignment is by numeric user ID, with no temporal semantics.
- Validation/test context windows and negatives are resampled by the dataset. Training Spearman/AUC concatenate up to the last ten batches; validation/test concatenate all loader batches. Spearman uses only nonpadding positive targets. For hurdle models, ranking metrics use `sigmoid(play_logits)`, not the regression head. This is not mean per-listener correlation or a temporal discovery benchmark. Rank-percentile scores assign distinct ranks to ties.
- The negative sampler rejects only prepared-library members, allows duplicate negatives, and has no termination guard if the entire catalog is excluded.
- The harness run cache omits `top_k`; encoded-library and run caches do not include artifact modification times. Replaced artifacts or changed requests can yield stale results. Thresholds are quantized in cache keys.
- `pickle.load` and `torch.load(..., weights_only=False)` can execute code from untrusted artifacts. Other `torch.load` calls depend on the installed library's defaults. “Local” is not a trust guarantee.
- The embedder enables `trust_remote_code=True` without pinning a model revision. It can fetch and execute external code; no weights or remote code were downloaded for this release.
- The hyperparameter sweep can prune checkpoint files; other scripts overwrite or append to data, checkpoints, and logs. Progress and dry-run paths must not be assumed safe or read-only from their names.
- Some source comments retain obsolete dimensions, sequence-model promises, historical estimates, and intended-effect claims. They are historical annotations; the current definitions and the qualifications in these documents take precedence.

[requirements.txt](../requirements.txt) is retained as historical evidence, with broad lower bounds rather than a tested lock. `train.py` imports SciPy, which is not directly declared there. Other external/transitive requirements and compatibility were not resolved. The omitted root npm manifest declared only old `codex` tooling unrelated to this harness. No dependencies were modernized or installed, and no runtime success is claimed.
