# DoDiligence accuracy results

232 citations: 218 that US courts found to be fake, misquoted or misrepresented in real filings, and 14 correct citations written by judges (taken from the opinions in the test set).

| | Result | 95% CI |
|---|---|---|
| Problems caught (recall) | 94.0% (205/218) | 90.1%–96.5% |
| Flags that were real problems (precision) | 99.5% (205/206) | 97.3%–99.9% |
| F1 | 0.967 | |

## By error type

| Group | Items | Caught | Right error type | Sent to human review | Missed |
|---|---|---|---|---|---|
| Fake or wrong-case citations | 158 | 97.5% | 154 | 4 | 0 |
| Invented quotations | 15 | 93.3% | 14 | 1 | 0 |
| Real case, misstated holding | 45 | 82.2% | 35 | 6 | 2 |

| Correct citations | Items | Passed | False alarms | Sent to human review |
|---|---|---|---|---|
| Correct citations written by judges | 14 | 92.9% | 1 | 0 |

## Which check caught each error

The case lookup and quote match are plain checks; only the support check uses a language model, and only on citations that pass the other two.

| Group | Case lookup | Quote match | Support check | Sent to review | Missed |
|---|---|---|---|---|---|
| Fake or wrong-case citations | 154 | 0 | 0 | 4 | 0 |
| Invented quotations | 0 | 14 | 0 | 1 | 0 |
| Real case, misstated holding | 2 | 3 | 32 | 6 | 2 |

With the support check switched off, the lookup and quote checks alone catch 11.1% of misstated holdings.

## Support check

53 language-model calls (claude-sonnet-5-5), averaging 3,089 input and 127 output tokens per check: each check reads about five passages, not the whole opinion. Everything else runs locally on the CPU; see `python -m eval.perf`.

## Status counts

| Group | VERIFIED | QUOTE_OK | NOT_FOUND | NAME_MISMATCH | MISQUOTED | CONTRADICTED | UNSUPPORTED | REVIEW |
|---|---|---|---|---|---|---|---|---|
| Fake or wrong-case citations | 0 | 0 | 37 | 117 | 0 | 0 | 0 | 4 |
| Invented quotations | 0 | 0 | 0 | 0 | 14 | 0 | 0 | 1 |
| Real case, misstated holding | 2 | 0 | 1 | 1 | 3 | 3 | 29 | 6 |
| Correct citations written by judges | 13 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
