"""Ball-level Pressure Index (PI, 0-100).

All four components are computed from *match state before the ball is bowled*:

Stage   = (balls_bowled / 120)^2                      late overs matter more (convex)
Wickets = min(wickets_lost / WKT_SATURATION, 1)       batting side in trouble
Deficit = runs behind par pace
            1st inns: par_cum(b)   - score
            2nd inns: runs_needed  - par_remaining(b)
          (par curves are measured from the data itself, per format)
Close   = 1 - min(|Deficit| / CLOSE_SCALE, 1)         contest is on a knife-edge, in either direction
Demand  = min(max(Deficit, 0) / DEMAND_SCALE, 1)      batting side behind par -> must take risk

PI = 100 * (w_close*Close + w_stage*Stage + w_wkt*Wickets + w_demand*Demand)

Because Close and Stage multiply nothing but add, a ball can be high-pressure for two reasons:
a tight late finish, or a collapse / big deficit. Both batter and bowler experience the same PI on a
given ball (it is a property of the match situation, not of a side).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import config


def par_curves(df: pd.DataFrame) -> dict:
    """Mean cumulative score before ball b and mean runs remaining, from 1st innings, by tournament."""
    curves = {}
    first = df[df["innings"] == 1]
    for t, d in first.groupby("tournament"):
        tot = d.groupby("match_id")["runs_total"].sum()
        # score at start of each legal-ball index
        idx = np.arange(120)
        mean_cum = np.zeros(120)
        per_match = d.groupby("match_id")
        cum = np.zeros((len(tot), 120))
        for i, (mid, g) in enumerate(per_match):
            b = g["balls_before"].values
            s = g["score_before"].values
            # last score_before seen at or before each ball index (forward fill)
            arr = np.full(120, np.nan)
            arr[b] = s
            arr = pd.Series(arr).ffill().fillna(0).values
            cum[i] = arr
        mean_cum = cum.mean(axis=0)
        mean_total = tot.mean()
        curves[t] = dict(cum=mean_cum, rem=np.maximum(mean_total - mean_cum, 0))
    return curves


def add_pressure(df: pd.DataFrame, weights=None, close_scale=None, demand_scale=None, wkt_sat=None) -> pd.DataFrame:
    w = weights or config.PI_WEIGHTS
    cs = close_scale or config.CLOSE_SCALE
    ds = demand_scale or config.DEMAND_SCALE
    ws = wkt_sat or config.WKT_SATURATION
    df = df.copy()
    curves = par_curves(df)

    totals1 = df[df["innings"] == 1].groupby("match_id")["runs_total"].sum()
    df["target"] = df["match_id"].map(totals1) + 1
    b = df["balls_before"].values
    par_cum = np.zeros(len(df)); par_rem = np.zeros(len(df))
    for t, c in curves.items():
        mk = (df["tournament"] == t).values
        par_cum[mk] = c["cum"][b[mk]]
        par_rem[mk] = c["rem"][b[mk]]

    needed = df["target"] - df["score_before"]
    deficit = np.where(df["innings"] == 1, par_cum - df["score_before"], needed - par_rem)
    df["deficit"] = deficit
    balls_left = 120 - df["balls_before"]
    df["runs_needed"] = np.where(df["innings"] == 2, needed, np.nan)
    df["rrr"] = np.where(df["innings"] == 2, needed / (balls_left / 6), np.nan)

    stage = (df["balls_before"] / 120.0) ** 2
    wk = np.clip(df["wkts_before"] / ws, 0, 1)
    close = 1 - np.clip(np.abs(deficit) / cs, 0, 1)
    demand = np.clip(np.maximum(deficit, 0) / ds, 0, 1)
    df["pi_stage"], df["pi_wkt"], df["pi_close"], df["pi_demand"] = stage, wk, close, demand
    df["PI"] = 100 * (w["close"] * close + w["stage"] * stage + w["wkt"] * wk + w["demand"] * demand)

    # chase already decided -> damp (target reached is impossible pre-ball; hopeless chase = RRR>>)
    hopeless = (df["innings"] == 2) & (df["rrr"] > 24) & (balls_left < 24)
    df.loc[hopeless, "PI"] *= 0.6
    return df


def label_pressure(df: pd.DataFrame, pctl: float | None = None) -> tuple[pd.DataFrame, float]:
    pctl = config.HIGH_PRESSURE_PCTL if pctl is None else pctl
    thr = float(np.percentile(df["PI"], pctl))
    df = df.copy()
    df["high_pressure"] = df["PI"] >= thr
    df["pi_bucket"] = pd.cut(df["PI"], [-1, thr * 0.5, thr * 0.8, thr, 101],
                             labels=["Low", "Moderate", "Elevated", "High"]).astype(str)
    return df, thr


def validate_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """Does high PI really behave differently? Outcome rates by PI decile."""
    d = df[df["legal"]].copy()
    d["decile"] = pd.qcut(d["PI"], 10, labels=False, duplicates="drop") + 1
    out = d.groupby("decile").agg(PI_mean=("PI", "mean"), runs_per_ball=("runs_total", "mean"),
                                  wicket_rate=("is_wicket", "mean"),
                                  dot_rate=("runs_total", lambda s: (s == 0).mean()),
                                  boundary_rate=("runs_bat", lambda s: (s >= 4).mean()),
                                  balls=("PI", "size")).reset_index()
    return out
