# Most Reliable Players Under Pressure — SupeRR Selector S2, Round 1
Pressure-index model → Streamlit dashboard → auto-generated methodology report. The ball-by-ball file is already in `data/`.

## Run it (pick one)
**A. One click** — Windows: double-click `run.bat` · macOS/Linux: `./run.sh`  → opens http://localhost:8501

**B. VS Code terminal**
```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py              # dashboard
python run_pipeline.py --report   # rankings CSVs + outputs/Methodology_Report.docx
```
**C. Docker** — `docker build -t rr-pressure . && docker run -p 8501:8501 rr-pressure`

**D. Streamlit Community Cloud (public link)** — push the folder to a **PRIVATE** GitHub repo (the RR data pack is proprietary; the
included `.gitignore` excludes `data/*.csv`, so remove that line only for a private repo), then share.streamlit.io → New app → `app.py`.

## What you get
Dashboard tabs: Executive summary · Pressure model · Batters · Bowlers · Player explorer · Squad fit & auction · Reliability · Submission.
All model settings (pressure weights, cut-off, wicket value, eligibility, filters) are in the sidebar; defaults live in `config.py`.

| file | role |
|---|---|
| `config.py` | every modelling parameter |
| `src/loader.py` | schema auto-detection, extras / wicket logic, joins |
| `src/pressure.py` | Pressure Index + validation |
| `src/metrics.py` | context-adjusted values, ratings, shrinkage, consistency, reliability, player profiles |
| `src/sensitivity.py` | top-10 stability |
| `src/report.py` | methodology .docx |
| `app.py` | dashboard |

## Submission checklist (deadline 18 Nov 2026, 11:59 PM IST)
1. `python run_pipeline.py --report` → open `outputs/Methodology_Report.docx`, add your name/team, edit in your own words.
2. Select the same 10 batters + 10 bowlers in the Unstop dropdowns (see `outputs/top10_*.csv` or the Submission tab).
3. Upload the report (or export it to PDF). Optionally add dashboard screenshots.
If you change anything in `config.py` or the sidebar, re-run the pipeline so report and rankings stay identical.
