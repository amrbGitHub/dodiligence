# DoDiligence: notes for Claude Code

A Streamlit demo that checks case citations in legal memos. It confirms the case exists and is the named case (CourtListener), that quotes match the opinion (rapidfuzz), and that the opinion supports the claim (Claude Haiku). README.md has the measured accuracy on real court cases.

## Goal of the handoff
1. Put this folder in a new **public** GitHub repo, `dodiligence`, on the `main` branch.
2. Deploy on Streamlit Community Cloud (share.streamlit.io) with entry point `app.py` and Python 3.12.

## Rules
- **Never commit keys.** `.streamlit/secrets.toml` is git-ignored. Keys go only in that local file and in Streamlit Cloud → App settings → Secrets:
  ```toml
  ANTHROPIC_API_KEY = "sk-ant-..."
  COURTLISTENER_TOKEN = "..."
  ```
  Before the first push, run `git grep -nE "sk-ant-|COURTLISTENER_TOKEN *= *\"[0-9a-f]{20,}"` and make sure it finds nothing.
- Don't edit `eval/data/*`. They are recorded CourtListener responses and benchmark results, and the Real cases and Accuracy tabs read them.
- Streamlit Cloud deployment is done by the user in the browser (it needs their GitHub sign-in). Give them the exact steps; don't try to automate it.

## Check locally before pushing
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # includes PaddleOCR on ONNX Runtime; to skip OCR, delete the last two lines
streamlit run app.py                                   # The Real cases and Accuracy tabs work with no keys
python -m eval.run_eval --no-ai                        # free sanity check of the benchmark
```

## If the Streamlit Cloud build runs out of memory
OCR is PaddleOCR (PP-OCRv6, small tier) on ONNX Runtime, about 420 MB at peak. Don't add `paddlepaddle` back: it has no Python 3.14 build. If memory is still a problem, delete the `paddleocr` and `onnxruntime` lines from `requirements.txt`, commit, and push. Only scanned-PDF/image uploads stop working, and the app says so.

## Layout
- `app.py`: UI
- `ui_accuracy.py`: Accuracy tab
- `core/`: pipeline
- `eval/`: benchmark (`python -m eval.run_eval`)
- `data/showcase.json`: which benchmark items the Real cases tab shows
