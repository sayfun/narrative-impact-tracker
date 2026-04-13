# Narrative Impact Tracker

**Empirically measure the influence of prediction market probability movements on media narratives.**

A research tool for computational journalism and platform studies. Connects live Polymarket probability data to GDELT media coverage, tests whether probability movements Granger-cause narrative shifts, and produces publication-ready statistical output alongside interactive visualisations.

**Live tool → [narrative-impact-tracker.streamlit.app](https://narrative-impact-tracker.streamlit.app)**

---

## What it does

Prediction markets like Polymarket don't just aggregate information — they generate and circulate stories about possible futures. This tool asks an empirical question: **do probability movements cause measurable changes in how media covers a story?**

For a given market (e.g. *"Will the US attack Iran before July 2026?"*), the tool:

1. Pulls daily WIN probability from the Polymarket CLOB API
2. Fetches GDELT media coverage intensity and article tone for matching topic terms
3. Extracts three narrative variables from article headlines:
   - **ERS** (Epistemic Register Score) — certainty vs. hedging language density
   - **PCF** (Probability Citation Frequency) — explicit probability language and market citations
   - **NCS** (Narrative Closure Score) — possibility-foreclosing language ("no path to victory", "effectively over")
4. Tests temporal relationships using Granger causality and event study analysis
5. Generates interactive charts and downloadable data for further analysis

---

## Theoretical grounding

The tool operationalises a three-mode enrollment framework for prediction markets as narrative machines:

| Mode | What it captures | Metric |
|------|-----------------|--------|
| **Financial enrollment** | Markets cited as authority in coverage | PCF |
| **Epistemic enrollment** | Shift from hedged to certain framing | ERS |
| **Narrative closure** | Probability as story-ending mechanism | NCS |

ERS and NCS extend the FRAME framework (Framework for Reflecting Agency in Media Environments) and its operationalised metrics — Framing Divergence Index (FDI) and Personalisation Intensity Index (PII) — to capture the specific narrative dynamics prediction markets introduce.

---

## Using the web app

Go to **[narrative-impact-tracker.streamlit.app](https://narrative-impact-tracker.streamlit.app)** — no installation or login required.

1. **Search** for a Polymarket market (e.g. `Iran attack`, `Fed rate cut`, `Ukraine ceasefire`)
2. **Select** the matching market from the dropdown — shows real market names, status, and 24h volume
3. **Set topic terms** for the GDELT coverage query (2–4 keywords work best)
4. **Set a date range** — use the window when the market was actively trading
5. **Run analysis** — results include all charts, statistical tests, and download buttons

---

## Installation (for researchers)

```bash
pip install narrative-impact-tracker
python -m spacy download en_core_web_sm
```

Or from source:
```bash
git clone https://github.com/sayfun/narrative-impact-tracker
cd narrative-impact-tracker
pip install -e .
python -m spacy download en_core_web_sm
```

### CLI usage

```bash
# Run full analysis and generate HTML report
narrative-tracker run \
  --market "presidential election winner 2024" \
  --topic Trump Harris election president \
  --start 2024-07-15 \
  --end 2024-11-07 \
  --output ./results

# Browse available markets
narrative-tracker markets --query "Iran attack"
```

### Python API

```python
from narrative_tracker import NarrativePipeline
from narrative_tracker.analysis import run_full_analysis

pipe = NarrativePipeline(
    market_query="presidential election winner 2024",
    topic_terms=["Trump", "Harris", "election"],
    start="2024-07-15",
    end="2024-11-07",
)
pipe.collect()

analysis = run_full_analysis(pipe.aligned, pipe.shocks)
print(pipe.summary())
```

---

## Statistical methods

**Granger causality tests** — tests whether lagged probability movements have predictive power over narrative variables above and beyond the series' own lags. Applied to first-differenced series after ADF stationarity testing. Maximum lag: 5 days. Both directions tested (probability → narrative and narrative → probability).

**Event study** — identifies sharp probability movements (≥8pp over 2 days) and measures pre/post changes in narrative variables within a ±5-day window. Closer to causal identification than Granger alone.

**Cross-correlation** — computes correlation between probability and coverage at lags −10 to +10 days. Identifies the lag at which probability-coverage co-movement peaks.

> **Important:** Granger causality ≠ structural causation. Both series may respond to a common third variable (debate, major news event) with different lag structures. For publication-quality results, manually classify each shock event as "clean" (no major scheduled event within ±72 hours) or "confounded" and report results separately.

---

## Data sources

| Source | What it provides | Cost |
|--------|-----------------|------|
| [Polymarket CLOB API](https://clob.polymarket.com) | Daily YES/NO probabilities | Free, public |
| [GDELT v2 Doc API](https://api.gdeltproject.org) | Coverage volume + article tone | Free, public |

**Coverage limitations:**
- GDELT returns headline metadata only — ERS/PCF/NCS values are computed from headlines, not full article text. Absolute values are suppressed relative to full-text analysis; time-series patterns are preserved. Validate against a human-coded sample before using as primary outcomes.
- GDELT skews toward English-language wire services. Under-represents Substack, financial media, and non-English coverage.
- Polymarket market search covers ~400 most actively traded markets. Niche or low-volume markets may not appear in search results.
- Polymarket coverage is strongest for US politics, macroeconomics, crypto, and geopolitics. Non-Anglophone political events are underrepresented.

---

## Output files

Running the CLI produces:

```
<output-dir>/
├── narrative_impact_report.html        ← self-contained HTML report
└── data/
    ├── aligned_frame.csv               ← main analytical dataset
    ├── probability_series.csv          ← raw Polymarket daily prices
    ├── articles.csv                    ← GDELT article metadata
    └── narrative_features_daily.csv    ← ERS, PCF, NCS daily aggregates
```

---

## Recommended markets for research

Markets with sufficient probability variance and media coverage for meaningful analysis:

| Query | Window | Notes |
|-------|--------|-------|
| `presidential election winner 2024` | Jul–Nov 2024 | Richest dataset; dense probability data |
| `Iran attack military 2026` | Jan–Apr 2026 | Active right now |
| `Ukraine ceasefire peace` | 2024–2025 | Long-running, high coverage |
| `Fed rate cut 2025` | 2024–2025 | Strong macro coverage |
| `Trump impeachment 2025` | 2025 | US politics |

---

## Citation

If you use this tool in published research, please cite:

```bibtex
@software{narrative_impact_tracker,
  author  = {Funk, Sascha H.},
  title   = {Narrative Impact Tracker: Empirically measuring prediction market influence on media narratives},
  year    = {2025},
  url     = {https://github.com/sayfun/narrative-impact-tracker},
  version = {1.0.0}
}
```

Or in-text: Funk, S. H. (2025). *Narrative Impact Tracker* [Software]. https://github.com/sayfun/narrative-impact-tracker

---

## Contributing

Pull requests welcome. Priority areas:
- MediaCloud integration for full-text feature extraction
- Support for additional prediction market platforms (Metaculus, Manifold)
- Probability Attribution Index (PAI) implementation
- Inter-rater reliability annotation interface for manual event coding

---

## Licence

MIT
