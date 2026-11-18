"""CLI:  python run_pipeline.py [--bbb path.csv] [--report]
Writes rankings, tables and (optionally) the methodology .docx into /outputs."""
import argparse, json
import pandas as pd
import config
from src import pipeline, metrics, pressure, sensitivity

ap = argparse.ArgumentParser()
ap.add_argument("--bbb", help="path to ball-by-ball CSV (default: auto-detect in /data)")
ap.add_argument("--report", action="store_true", help="also build outputs/Methodology_Report.docx")
a = ap.parse_args()

config.OUT_DIR.mkdir(exist_ok=True)
df, sup, info = pipeline.build_ball_table(bbb_path=a.bbb)
print(f"{info['n_matches']} matches, {info['n_balls']:,} balls | high-pressure threshold PI >= {info['threshold']:.1f}")
for n in info["notes"]:
    print("  note:", n)
tabs = pipeline.rank_tables(df, sup)
rel = {r: metrics.split_half_reliability(df, r) for r in ("bat", "bowl")}
sens, freq = {}, {}
for r in ("bat", "bowl"):
    sens[r], freq[r] = sensitivity.top10_stability(df, r)
    tabs[r].to_csv(config.OUT_DIR / f"ratings_{'batters' if r=='bat' else 'bowlers'}_full.csv", index=False)
    top = pipeline.top_n(tabs[r])
    top[["rank", "player", "latest_team", "hp_balls", "hp_matches", "PV", "resilience", "consistency", "score"]].round(3) \
        .to_csv(config.OUT_DIR / f"top10_{'batters' if r=='bat' else 'bowlers'}.csv", index=False)
    print(f"\nTOP 10 {'BATTERS' if r=='bat' else 'BOWLERS'}")
    print(top[["rank", "player", "latest_team", "hp_balls", "PV", "resilience", "consistency", "score"]].round(2).to_string(index=False))
    print("split-half reliability (Spearman):", rel[r])
pressure.validate_pressure(df).to_csv(config.OUT_DIR / "pressure_validation_by_decile.csv", index=False)
json.dump(dict(info={k: v for k, v in info.items() if k != "schema"}, schema=info["schema"], reliability=rel),
          open(config.OUT_DIR / "run_summary.json", "w"), indent=2, default=str)
if a.report:
    from src import report
    p = report.build_report(df, sup, info, tabs, rel, sens, freq)
    print("\nReport written:", p)
