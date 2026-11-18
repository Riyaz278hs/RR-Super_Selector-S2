import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import loader
raw = pd.read_csv(sys.argv[1], nrows=2000, low_memory=False)
print("Columns:", list(raw.columns), "\n")
for k, v in loader.detect_schema(raw.columns).items():
    print(f"{k:12s} -> {v}")
print("\nRequired: match_id, innings, batter, bowler, runs (bat) + extras/total, wicket info.")
