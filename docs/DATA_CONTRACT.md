# Data contract

## Canonical data scope

Use the matched [Zenodo PEMS-BAY release 4263971](https://zenodo.org/records/4263971), as pinned by [ADR-0006](DECISIONS/ADR-0006-dataset-provenance-and-zero-policy.md). Verify each required MD5 checksum, then record filenames, acquisition date, sensor/node ordering evidence, and license in a dataset manifest before a benchmark experiment. Raw data is excluded from Git.

## Raw and processed schemas

| Asset | Shape / schema | Rules |
|---|---|---|
| Time series | `[T, 325]` | ordered timestamps; one speed feature per sensor |
| Timestamp index | `[T]` | strictly increasing, no duplicates, exact five-minute intervals for eligible benchmark data |
| Adjacency | `[325, 325]` | finite; same node ordering as time-series columns |
| Windows `X` | `[S, L, 325, 1]` at loader boundary | `L=12` default |
| Targets `Y` | `[S, H, 325, 1]` at loader boundary | `H=12` default |
| Observation mask `M` | same data shape or documented broadcast form | `1=observed`, `0=withheld`; stored separately from values |

No processed NPZ or dataset class currently exists. Phase 3 will persist arrays in the canonical four-dimensional loader-boundary shape above; any storage-level compression or axis omission must be versioned in its manifest and restored explicitly rather than silently transposed.

## Preprocessing rules

1. Validate dimensions and graph alignment before transformations.
2. Split raw observations chronologically into 70% train, 10% validation, and 20% test.
3. Fit each per-sensor z-score mean and standard deviation only on the training partition.
4. Apply those frozen parameters to all partitions.
5. Generate windows within each split; no window crosses split boundaries (ADR-0003).
6. Store scaler parameters, configuration, and data manifest with every derived artefact.

## Missing values

There are two distinct concepts and they must never be merged:

- **Dataset-native missingness:** `0.0` and `NaN` values in the canonical source (ADR-0006). The `st_dssm.missingness.extract_native_missingness` function implements this policy, returning a raw array with `0.0` converted to `np.nan` and a boolean observation mask where `True` indicates a genuinely observed value.
- **Experimental masking:** values deliberately withheld from the model to simulate failure. Original values remain evaluation targets and must not be fed back to the model.

### Leakage-safe repair

Data repair is implemented in `st_dssm.imputation`. It follows a strict causal contract:
1. **No future information:** `apply_causal_forward_fill` strictly forward-fills missing values from previous observations.
2. **Explicit training fallbacks:** Leading gaps are filled using a per-sensor statistic fitted *strictly* on training data via `fit_fallback_statistics`.
3. **Immutability:** Imputation never mutates the original data array in place.
4. **Validation:** Imputation will fail if any unresolved NaNs remain.

## Data validation gate

An experiment may proceed only when timestamp cadence, source pairing, sensor IDs/order, zero/missing policy, and split/scaler metadata are all recorded. The automated CLI `st-dssm-validate` enforces this by:
1. Re-verifying canonical MD5 checksums.
2. Loading and validating the exact time-series schema, graph adjacency shape `[325, 325]`, and exact sensor ID alignment (`st_dssm.validator`).
3. Verifying that the index is a strict `DatetimeIndex` with no duplicates.
4. Rejecting irregular intervals, non-numeric data, and infinities.
5. Extracting missingness semantics and recording exactly how many `0.0` and `NaN` values occur in a deterministic JSON report.
