"""
cli.py
------
Command-line interface for the Narrative Impact Tracker.

Entry point: `narrative-tracker`

Subcommands
-----------
  run       Run the full pipeline and generate an HTML report
  markets   Browse and search Polymarket markets
  version   Show version and dependency info

Examples
--------
  # Full analysis with HTML report
  narrative-tracker run \\
    --market "presidential election winner 2024" \\
    --topic Trump Harris election president \\
    --start 2024-07-15 \\
    --end 2024-11-07 \\
    --output ./results

  # Quick run without article-level tone (faster)
  narrative-tracker run \\
    --market "UK general election 2024" \\
    --topic "Labour" "Tories" "Starmer" "election" \\
    --start 2024-01-01 --end 2024-07-05 \\
    --no-articles

  # Find markets to analyse
  narrative-tracker markets --query "general election 2024" --limit 10
"""

import argparse
import sys
import textwrap
import os
import json
from pathlib import Path
from datetime import datetime


def _print_banner():
    print("\n" + "="*60)
    print("  Narrative Impact Tracker")
    print("  Prediction markets as narrative machines — empirically.")
    print("="*60 + "\n")


# ── `run` subcommand ──────────────────────────────────────────────────────────

def cmd_run(args):
    from narrative_tracker.pipeline  import NarrativePipeline
    from narrative_tracker.analysis  import run_full_analysis
    from narrative_tracker.features  import aggregate_daily_features, extract_headline_features_df
    from narrative_tracker.report    import generate_report, save_report
    from narrative_tracker.gdelt     import build_prediction_market_query

    _print_banner()

    # Resolve output directory
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(exist_ok=True)

    print(f"Output directory : {out_dir}")
    print(f"Market query     : {args.market}")
    print(f"Topic terms      : {', '.join(args.topic)}")
    print(f"Analysis window  : {args.start} → {args.end or 'today'}")
    print(f"Fetch articles   : {'yes' if not args.no_articles else 'no (--no-articles flag)'}")
    print()

    # ── Collect data ──
    pipe = NarrativePipeline(
        market_query    = args.market,
        topic_terms     = args.topic,
        start           = args.start,
        end             = args.end,
        market_index    = args.market_index,
        shock_threshold = args.shock_threshold,
        fetch_articles  = not args.no_articles,
        article_window_days = args.article_window,
    )

    try:
        pipe.collect(verbose=True)
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        print("Try: narrative-tracker markets --query '<your search term>'")
        sys.exit(1)

    print()
    print(pipe.summary())
    print()

    # ── Save raw data ──
    aligned_path = data_dir / "aligned_frame.csv"
    prob_path    = data_dir / "probability_series.csv"
    pipe.aligned.to_csv(aligned_path, index=False)
    pipe.prob_df.to_csv(prob_path,    index=False)
    print(f"Data saved: {aligned_path}")
    print(f"Data saved: {prob_path}")

    if pipe.articles_df is not None and not pipe.articles_df.empty:
        articles_path = data_dir / "articles.csv"
        pipe.articles_df.to_csv(articles_path, index=False)
        print(f"Data saved: {articles_path}")

    # ── Narrative features ──
    features_daily = None
    if pipe.articles_df is not None and not pipe.articles_df.empty:
        print("\nExtracting narrative features …")
        enriched = extract_headline_features_df(pipe.articles_df)
        features_daily = aggregate_daily_features(enriched)
        feats_path = data_dir / "narrative_features_daily.csv"
        features_daily.to_csv(feats_path, index=False)
        print(f"Data saved: {feats_path}")

        # Merge features into aligned frame for analysis
        if features_daily is not None and not features_daily.empty:
            import pandas as pd
            feat_cols = [c for c in features_daily.columns if c != "date"]
            feat_merge = features_daily[["date"] + feat_cols].copy()
            feat_merge["date"] = pd.to_datetime(feat_merge["date"], utc=True).dt.normalize()
            aligned_w_feats = pipe.aligned.copy()
            aligned_w_feats["date"] = pd.to_datetime(aligned_w_feats["date"], utc=True).dt.normalize()
            aligned_w_feats = aligned_w_feats.merge(feat_merge, on="date", how="left")
        else:
            aligned_w_feats = pipe.aligned
    else:
        aligned_w_feats = pipe.aligned

    # ── Statistical analysis ──
    print("\nRunning statistical analysis …")
    try:
        analysis = run_full_analysis(aligned_w_feats, pipe.shocks, verbose=True)
    except Exception as e:
        print(f"  Warning: analysis failed ({e}). Report will be generated without stats.")
        analysis = {"granger": {}, "xcorr": {}, "events": {}}

    # ── Generate HTML report ──
    print("\nGenerating HTML report …")
    gdelt_query = build_prediction_market_query(args.topic[:4])

    html = generate_report(
        aligned_df       = aligned_w_feats,
        shocks_df        = pipe.shocks,
        analysis_results = analysis,
        market_question  = pipe.market_meta["question"],
        gdelt_query      = gdelt_query,
        start            = args.start,
        end              = args.end or datetime.utcnow().strftime("%Y-%m-%d"),
        features_daily   = features_daily,
    )

    report_path = out_dir / "narrative_impact_report.html"
    save_report(html, str(report_path))

    has_articles = pipe.articles_df is not None and not pipe.articles_df.empty
    # ── Done ──
    print(f"""
{'='*60}
Done.

Outputs:
  HTML report   : {report_path}
  Data folder   : {data_dir}/
      aligned_frame.csv          <- main analytical dataset
      probability_series.csv     <- raw Polymarket daily prices
      {'articles.csv                 <- GDELT article metadata' if has_articles else '(articles not fetched — rerun without --no-articles)'}
      {'narrative_features_daily.csv <- ERS, PCF, NCS daily aggregates' if features_daily is not None else ''}

Open the HTML report in any browser — it is fully self-contained.
{'='*60}
""")


# ── `markets` subcommand ──────────────────────────────────────────────────────

def cmd_markets(args):
    from narrative_tracker.polymarket import search_markets

    print(f"\nSearching Polymarket: '{args.query}' …\n")
    markets = search_markets(args.query, limit=args.limit)

    if markets.empty:
        print("No markets found.")
        return

    valid = markets[markets["token_ids"].apply(lambda x: len(x) > 0)]
    print(f"Found {len(valid)} markets with CLOB data:\n")

    for i, (_, row) in enumerate(valid.head(args.limit).iterrows()):
        status = "ACTIVE" if row.get("active") else "CLOSED"
        print(f"  [{i:2d}] [{status}] {row['question'][:65]}")
        print(f"        Volume: ${row['volume']:>14,.0f}")
        print(f"        YES token (last 12): …{row['token_ids'][0][-12:]}")
        print()

    print("To use a specific market, pass --market-index N to `narrative-tracker run`.")
    print("Example:")
    print(f"  narrative-tracker run --market '{args.query}' --market-index 0 \\")
    print(f"    --topic term1 term2 term3 --start YYYY-MM-DD")


# ── `version` subcommand ─────────────────────────────────────────────────────

def cmd_version(args):
    from narrative_tracker import __version__
    import platform

    print(f"\nnarrative-impact-tracker {__version__}")
    print(f"Python {platform.python_version()} on {platform.system()}")

    deps = ["requests", "pandas", "spacy", "statsmodels", "matplotlib", "numpy"]
    for dep in deps:
        try:
            import importlib
            mod = importlib.import_module(dep)
            ver = getattr(mod, "__version__", "?")
            print(f"  {dep:<15} {ver}")
        except ImportError:
            print(f"  {dep:<15} NOT INSTALLED")

    try:
        import spacy
        models = spacy.util.get_installed_models()
        print(f"\nspaCy models: {', '.join(models) if models else 'none installed'}")
        if "en_core_web_sm" not in models:
            print("  Run: python -m spacy download en_core_web_sm")
    except Exception:
        pass


# ── argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="narrative-tracker",
        description=(
            "Narrative Impact Tracker — empirically measure prediction market "
            "influence on media narratives."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              narrative-tracker run \\
                --market "presidential election winner 2024" \\
                --topic Trump Harris election president \\
                --start 2024-07-15 --end 2024-11-07

              narrative-tracker markets --query "UK general election 2024"
              narrative-tracker version
        """),
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── run ──
    run_p = sub.add_parser("run", help="Run pipeline and generate HTML report")

    run_p.add_argument(
        "--market", required=True,
        help='Market search query (e.g. "presidential election winner 2024")',
    )
    run_p.add_argument(
        "--topic", nargs="+", required=True,
        help="Topic terms for GDELT query (e.g. Trump Harris election)",
    )
    run_p.add_argument(
        "--start", required=True,
        help="Analysis start date (YYYY-MM-DD)",
    )
    run_p.add_argument(
        "--end", default=None,
        help="Analysis end date (YYYY-MM-DD). Defaults to today.",
    )
    run_p.add_argument(
        "--output", default="./narrative_tracker_output",
        help="Output directory (default: ./narrative_tracker_output)",
    )
    run_p.add_argument(
        "--market-index", type=int, default=0, dest="market_index",
        help="Which search result to use (0 = highest volume). Default: 0",
    )
    run_p.add_argument(
        "--shock-threshold", type=float, default=0.08, dest="shock_threshold",
        help="Minimum probability movement to flag as sharp shock (default: 0.08 = 8pp)",
    )
    run_p.add_argument(
        "--no-articles", action="store_true",
        help="Skip GDELT article-list fetching (faster, no tone/feature data)",
    )
    run_p.add_argument(
        "--article-window", type=int, default=7, dest="article_window",
        help="GDELT pagination window in days (default: 7)",
    )

    # ── markets ──
    mkt_p = sub.add_parser("markets", help="Search and browse Polymarket markets")
    mkt_p.add_argument("--query", required=True, help="Search query")
    mkt_p.add_argument("--limit", type=int, default=10, help="Number of results (default: 10)")

    # ── version ──
    sub.add_parser("version", help="Show version and dependency info")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "markets":
        cmd_markets(args)
    elif args.command == "version":
        cmd_version(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
