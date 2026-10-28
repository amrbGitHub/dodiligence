# DoDiligence

**Check every case citation before you file.**

Checks the case citations in a legal memo or brief, the way a careful associate would:

1. **Does the case exist?** Citations are found with [eyecite](https://github.com/freelawproject/eyecite) and looked up on [CourtListener](https://www.courtlistener.com). If the citation leads to a *different* case than the one named, that's flagged too, which is how most AI-invented citations show up.
2. **Is the quote real?** Every quotation is fuzzy-matched against the opinion text, so curly quotes, brackets and OCR noise don't cause false alarms, but changed words do. Quotes the memo attributes to a statute or contract aren't held against the case.
3. **Does the case say that?** Claude Haiku reads only passages from the cited opinion and decides whether they support, contradict, or don't support the memo's proposition.
4. **Scanned filings** are read with PaddleOCR (PP-OCRv6) on the server, running on ONNX Runtime. A page is OCR'd only if it has no real text layer, so court e-filing stamps on scanned pages don't hide the scan. Five scanned pages take about 25 seconds on two CPU cores, in about 420 MB of memory.

## Accuracy (measured on real court cases)

Ground truth comes from courts, not from us. We took **218 citations that US judges found to be fake, misquoted or misrepresented** in real filings. They come from Damien Charlotin's [AI Hallucination Cases database](https://www.damiencharlotin.com/hallucinations/) (CC BY 4.0) and the *Mata v. Avianca* sanctions order. To those we added correct citations written by judges.

| Kind of error | Cases | Flagged | Without the Claude step |
|---|---|---|---|
| Fake case, or citation leads to a different case | 158 | **97.5%** (other 2.5% sent to human review) | 97.5% |
| Invented quotation | 15 | **93.3%** | 93.3% |
| Real case cited for something it doesn't say | 45 | **95.6%** | 11.1% |
| **All errors** | **218** | **96.8%** (95% CI 93.5–98.4%) | |

| Correct citations written by judges | Cases | Passed | Wrongly flagged |
|---|---|---|---|
| | 14 | 12 | 2 |

- **What Claude adds.** Existence and quote checks catch almost every fabricated citation without any AI. Misstated holdings are the hard, high-stakes part, and there the Claude step goes from 11% to 96%.
- **Fast and light.** Finding citations, matching quotes and picking passages take a median of a few milliseconds per citation on one CPU core, in about 130 MB of memory, with no GPU. The only model call is the support check: `claude-haiku-5-5` reads about five passages (roughly 1,900 tokens), not the whole opinion, and answers in about a second. Measured with `python -m eval.perf`.
- **Stable.** Two fresh runs from scratch gave identical results.
- **Honest limits.**
  - The correct-citation set is small (14), and it was looked at while the rules were tuned, so treat 12/14 as optimistic. Of the two false alarms, one is a judge quoting a statute through a case, and one is a broad "see, e.g." cite. A larger held-out set (63 sentences, already selected) is waiting on CourtListener's free-tier limit of 50 requests an hour.
  - For misstated holdings, the filer's proposition was reconstructed from the court's description of it, so it isn't always word-for-word what was filed.
  - Only the majority opinion of each case is checked.

Every test case, including the misses, is in the app's **Accuracy** tab and in `eval/data/`. Full table: [`eval/RESULTS.md`](eval/RESULTS.md).

### Re-run it

```bash
ANTHROPIC_API_KEY=sk-ant-... python -m eval.run_eval      # full run
python -m eval.run_eval --no-ai                           # existence + quote checks only, no key needed
```

CourtListener responses are recorded in `eval/data/recording.json.gz`, so every run sees the same opinions. Claude verdicts are cached in `eval/data/judgments.json`; delete that file to redo the AI step.

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

- **Claude:** console.anthropic.com → API keys. Use a key just for this demo.
- **CourtListener:** free account → Profile → Developer Tools → API token. New accounts get **50 requests an hour** (5 a minute). A memo uses one request for the lookup plus one per case, so the live "Paste a memo" tab handles roughly 8 short memos an hour; DoDiligence waits and retries when throttled.
- The public demo caps new AI checks at 3 per visitor session and 150 per day (`RUNS_PER_SESSION`, `RUNS_PER_DAY`).

## Project layout

```
app.py              Streamlit UI (Sample memo, Real cases, Accuracy, Paste, Upload)
ui_accuracy.py      Accuracy tab
core/extract.py     find citations (merging parallel cites), claims, and quotes
core/sources.py     CourtListener API, with recorded replays
core/match.py       quote matching (rapidfuzz) and passage retrieval (BM25)
core/judge.py       Claude: does the opinion support the claim?
core/pipeline.py    ties it together, one verdict per citation
core/ocr.py         PDF text layer, or PP-OCRv6 for scans and images
eval/run_eval.py    the benchmark
eval/perf.py        speed and memory (writes data/perf.json)
eval/data/          test items, recorded CourtListener responses, results
data/showcase.json  which benchmark items the Real cases tab shows
data/sample_memo.txt the Sample memo tab (the six fake cases from Mata v. Avianca)
```

Error cases: Damien Charlotin, *AI Hallucination Cases* database, CC BY 4.0. Opinion text: CourtListener (Free Law Project). This is a demo, not legal advice.
