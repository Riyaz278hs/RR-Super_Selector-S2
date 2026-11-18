"""End-to-end orchestration used by both the CLI and the dashboard."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import config
from . import loader
from . import metrics, pressure


def build_ball_table(bbb_path=None, raw=None, schema=None, data_dir=None, pi_weights=None, pctl=None):
    sup = loader.load_supporting(data_dir)
    if raw is None:
        bbb_path = bbb_path or loader.find_bbb_file(data_dir)
        if bbb_path is None:
            raise FileNotFoundError("No ball-by-ball CSV found in /data. Copy it there (see README).")
        raw = pd.read_csv(bbb_path, low_memory=False)
    schema = schema or loader.detect_schema(raw.columns)
    df, info = loader.normalise_bbb(raw, schema, sup)
    df = pressure.add_pressure(df, weights=pi_weights)
    df, thr = pressure.label_pressure(df, pctl)
    df = metrics.add_expectations(df)
    info.update(schema=schema, threshold=thr, n_balls=len(df), n_matches=df["match_id"].nunique())
    return df, sup, info


def rank_tables(df, sup, **kw):
    meta = metrics.player_meta(sup)
    if config.REQUIRE_ACTIVE_2026 and "restrict_ids" not in kw:
        kw["restrict_ids"] = set(meta.index[meta["active_2026"]])
    out = {}
    prof = {"bat": metrics.batter_profile(df), "bowl": metrics.bowler_profile(df)}
    for role in ("bat", "bowl"):
        t = metrics.player_ratings(df, role, **kw)
        t = t.merge(meta, left_on="player_id", right_index=True, how="left")
        t = t.merge(prof[role], left_on="player_id", right_index=True, how="left")
        t["fit"] = t.apply(lambda r: metrics.fit_text(r, role) if r["eligible"] else "", axis=1)
        out[role] = t
    return out


def top_n(t, n=10):
    return t[t["eligible"]].head(n)
