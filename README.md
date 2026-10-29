# DoDiligence

**Check every case citation before you file.**

Checks the case citations in a legal memo or brief, the way a careful associate would. The checks run in order, and each citation stops at the first one it fails:

1. **Does the case exist, and is it the case named?** Citations are found with [eyecite](https://github.com/freelawproject/eyecite) and looked up on [CourtListener](https://www.courtlistener.com). A citation that leads to a *different* case than the one named is flagged too; that is how most invented citations show up.
2. **Are the quotes real?** Every quotation is fuzzy-matched against the opinion text, so curly quotes, brackets and OCR noise don't cause false alarms, but changed words do. Quotes the memo attributes to a statute or contract aren't held against the case.
3. **Does the opinion support the claim?** This is the only step that uses a language model. It sees the memo's proposition and about five passages picked from the opinion (BM25), not the whole opinion and not its own memory of the case, and answers supported, contradicted, not supported, or unclear. Unclear goes to a person as "Needs review".

Steps 1 and 2 are deterministic and run locally. **Scanned filings** are read first with PaddleOCR (PP-OCRv6) on ONNX Runtime; only pages without a real text layer are OCR'd, so court e-filing stamps don't hide a scan.

## Accuracy (measured on real court cases)

Ground truth comes from courts, not from us: **218 citations that US judges found to be fake, misquoted or misrepresented** in real filings, from Damien Charlotin's [AI Hallucination Cases database](https://www.damiencharlotin.com/hallucinations/) (CC BY 4.0) and the *Mata v. Avianca* sanctions order, plus correct citations written by judges.

| Kind of error | Cases | Flagged | Caught by |
|---|---|---|---|
| Fake case, or citation leads to a different case | 158 | **97.5%** | case lookup 154 · 4 sent to review |
| Invented quotation | 15 | **93.3%** | quote match 14 · 1 sent to review |
| Real case cited for something it doesn't say | 45 | **95.6%** | support check 38, quote match 3, case lookup 2 · 1 sent to review, 1 missed |
| **All errors** | **218** | **96.8%** (95% CI 93.5–98.4%) | |

| Correct citations written by judges | Cases | Passed | Wrongly flagged |
|---|---|---|---|
| | 14 | 12 | 2 |

- **Most problems need no AI.** 173 of the 211 catches come from the lookup and quote checks. The model is only asked about citations that survive them (71 of the 232 test citations), and that is where it matters: misstated holdings are the hard, high-stakes errors, and the deterministic checks alone catch 5 of 45.
- **Fast and light.** Finding citations, matching quotes and picking passages take a few milliseconds per citation on one CPU core, in about 130 MB of memory, with no GPU. A support check sends about 1,900 tokens and returns in about a second. Five scanned pages take about 25 seconds to OCR on two CPU cores, in about 420 MB. Measured with `python -m eval.perf`.
- **Stable.** Two fresh runs from scratch gave identical results.
- **Honest limits.**
  - The correct-citation set is small (14), and it was looked at while the rules were tuned, so treat 12/14 as optimistic. Of the two false alarms, one is a judge quoting a statute through a case, and one is a broad "see, e.g." cite. A larger held-out set (63 sentences, already selected) is waiting on CourtListener's free-tier request limits.
  - For misstated holdings, the filer's proposition was reconstructed from the court's description of it, so it isn't always word-for-word what was filed.
  - Only the majority opinion of each case is checked.

Every test case, including the misses, is in the app's **Accuracy** tab and in `eval/data/`. Full table: [`eval/RESULTS.md`](eval/RESULTS.md).

### Re-run it

```bash
ANTHROPIC_API_KEY=sk-ant-... python -m eval.run_eval      # full run
python -m eval.run_eval --no-ai                           # lookup + quote checks only, no key needed
```

CourtListener responses are recorded in `eval/data/recording.json.gz`, so every run sees the same opinions. Support-check verdicts are cached in `eval/data/judgments.json`; delete that file to redo that step.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then fill in your keys
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a **public** GitHub repo. `.streamlit/secrets.toml` is git-ignored, so keys stay off GitHub.
2. At [share.streamlit.io](https://share.streamlit.io): **Create app** → pick the repo, branch `main`, file `app.py`.
3. **Advanced settings** → any Python from 3.11 up (3.12–3.14 tested for installation), and paste your secrets:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   COURTLISTENER_TOKEN = "..."
   ```
4. **Deploy.** The first scanned upload downloads the PP-OCRv6 models (about 30 seconds, once per server start). If the app ever runs out of memory, delete the `paddleocr` and `onnxruntime` lines from `requirements.txt`; everything except scanned uploads keeps working.

The **Sample memo**, **Real cases** and **Accuracy** tabs run from recorded data and cached verdicts. They're instant and don't touch either API, so they work even without keys.

## Keys and limits

- **Language model:** the support check calls the Anthropic API (`CLAUDE_MODEL`, default `claude-haiku-5-5`). Key: console.anthropic.com → API keys; use one just for this demo.
- **CourtListener:** free account → Profile → Developer Tools → API token. Free accounts have small hourly **and daily** limits (this demo's account: 125 requests a day). A memo uses one request for the lookup plus one to three per case, so the live tabs handle only a handful of new memos a day. To stretch that, CourtListener responses are cached in memory and shared by all visitors, so re-checking a memo or citing a case already looked up costs nothing; a short throttle is retried, and a daily one is reported with the time until lookups work again. For a busier demo, ask Free Law Project for a higher limit.
- The public demo caps new support checks at 3 per visitor session and 150 per day (`RUNS_PER_SESSION`, `RUNS_PER_DAY`).

## Project layout

```
app.py              Streamlit UI (Sample memo, Real cases, Accuracy, Paste, Upload)
ui_accuracy.py      Accuracy tab
core/extract.py     find citations (merging parallel cites), claims, and quotes
core/sources.py     CourtListener API, with recorded replays
core/match.py       quote matching (rapidfuzz) and passage retrieval (BM25)
core/judge.py       support check: does the opinion support the claim? (language model)
core/pipeline.py    ties it together, one verdict per citation
core/ocr.py         PDF text layer, or PP-OCRv6 for scans and images
eval/run_eval.py    the benchmark
eval/perf.py        speed and memory (writes data/perf.json)
eval/data/          test items, recorded CourtListener responses, results
data/showcase.json  which benchmark items the Real cases tab shows
data/sample_memo.txt the Sample memo tab (the six fake cases from Mata v. Avianca)
```

Error cases: Damien Charlotin, *AI Hallucination Cases* database, CC BY 4.0. Opinion text: CourtListener (Free Law Project). This is a demo, not legal advice.
