"""Central configuration. Every modelling choice lives here so the report,
the dashboard and the CLI all use identical numbers."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

# ---- supporting file names (put them in /data) ----
F_COMPETITIONS = "competitions.csv"
F_MATCHES = "matches.csv"
F_PLAYING11 = "playing_eleven.csv"
F_MAPPING = "competition_team_player_mapping.csv"
BBB_CANDIDATES = ["ball_by_ball.csv", "ballbyball.csv", "bbb.csv", "deliveries.csv"]  # or any other csv: see loader

# ---- pressure index (0-100) ----
# PI = 100 * (W_CLOSE*Close + W_STAGE*Stage + W_WKT*Wickets + W_DEMAND*Demand)
PI_WEIGHTS = dict(close=0.35, stage=0.30, wkt=0.20, demand=0.15)
CLOSE_SCALE = 25.0     # runs of deficit-vs-par at which contest is "not close any more"
DEMAND_SCALE = 25.0    # runs behind par at which Demand saturates
WKT_SATURATION = 8     # wickets lost at which the wicket component saturates
HIGH_PRESSURE_PCTL = 65  # a ball is "high pressure" if PI >= this percentile of all balls (top 35%)

# ---- performance model ----
WICKET_VALUE_RUNS = 10.0   # runs a wicket is worth. Chosen on split-half reliability: wickets are rare/noisy, 10 beats 20 (see report)
SHRINK_BALLS_BAT = 60      # pseudo-balls of league-average performance (empirical-Bayes shrinkage)
SHRINK_BALLS_BOWL = 48
CONSIST_PRIOR = 4          # pseudo-matches (half positive) in consistency rate
MIN_PRESSURE_BALLS_PER_MATCH = 4

# ---- eligibility ----
MIN_PRESSURE_BALLS_BAT = 100
MIN_PRESSURE_BALLS_BOWL = 90
MIN_PRESSURE_MATCHES = 10

# ---- final score weights ----
SCORE_WEIGHTS = dict(value=0.50, resilience=0.20, consistency=0.30)

# Exclude matches with rain-revised overs / targets (pressure maths assume 20 overs)
EXCLUDE_REVISED = True

# Auction relevance: only rank players who appear in a 2026 squad (IPL 2026 or 2026 T20I series)
REQUIRE_ACTIVE_2026 = True
