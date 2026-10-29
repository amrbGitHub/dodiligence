# DoDiligence accuracy results

232 citations: 218 that US courts found to be fake, misquoted or misrepresented in real filings, and 14 correct citations written by judges (taken from the opinions in the test set).

| | Result | 95% CI |
|---|---|---|
| Problems caught (recall) | 96.8% (211/218) | 93.5%–98.4% |
| Flags that were real problems (precision) | 99.1% (211/213) | 96.6%–99.7% |
| F1 | 0.979 | |

## By error type

| Group | Items | Caught | Right error type | Sent to human review | Missed |
|---|---|---|---|---|---|
| Fake or wrong-case citations | 158 | 97.5% | 154 | 4 | 0 |
| Invented quotations | 15 | 93.3% | 14 | 1 | 0 |
| Real case, misstated holding | 45 | 95.6% | 41 | 1 | 1 |

| Correct citations | Items | Passed | False alarms | Sent to human review |
|---|---|---|---|---|
| Correct citations written by judges | 14 | 85.7% | 2 | 0 |

## Which check caught each error

The case lookup and quote match are plain checks; only the support check uses a language model, and only on citations that pass the other two.

| Group | Case lookup | Quote match | Support check | Sent to review | Missed |
|---|---|---|---|---|---|
| Fake or wrong-case citations | 154 | 0 | 0 | 4 | 0 |
| Invented quotations | 0 | 14 | 0 | 1 | 0 |
| Real case, misstated holding | 2 | 3 | 38 | 1 | 1 |

With the support check switched off, the lookup and quote checks alone catch 11.1% of misstated holdings.

## Support check

71 language-model calls (claude-haiku-5-5), averaging 1,925 input and 144 output tokens per check: each check reads about five passages, not the whole opinion. Everything else runs locally on the CPU; see `python -m eval.perf`.

## Status counts

| Group | VERIFIED | QUOTE_OK | NOT_FOUND | NAME_MISMATCH | MISQUOTED | CONTRADICTED | UNSUPPORTED | REVIEW |
|---|---|---|---|---|---|---|---|---|
| Fake or wrong-case citations | 0 | 0 | 37 | 117 | 0 | 0 | 0 | 4 |
| Invented quotations | 0 | 0 | 0 | 0 | 14 | 0 | 0 | 1 |
| Real case, misstated holding | 1 | 0 | 1 | 1 | 3 | 3 | 35 | 1 |
| Correct citations written by judges | 12 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
