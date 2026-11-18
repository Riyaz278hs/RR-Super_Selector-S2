"""Context-adjusted performance values and player ratings.

Idea: a player's pressure performance is judged against what an *average player* produces in the same
context (format, innings, over, wickets-down bucket), not against raw averages -- so a death-overs
hitter is not rewarded merely for batting at the death.

Batter value / ball  = (runs - E[runs]) - L * (out - E[out])
Bowler value / ball  = (E[runs] - runs conceded) + L * (bowler_wkt - E[bowler_wkt])
L = WICKET_VALUE_RUNS.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import config

CELL = ["tournament", "innings", "over_idx", "wbucket"]
FALLBACK = ["tournament", "innings", "phase", "wbucket"]


def add_expectations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["wbucket"] = pd.cut(df["wkts_before"], [-1, 2, 4, 10], labels=["0-2", "3-4", "5+"]).astype(str)

    def expect(src_mask, cols_map):
        base = df[src_mask]
        res = {}
        for name, col in cols_map.items():
            fine = base.groupby(CELL)[col].agg(["mean", "size"]).rename(columns={"mean": "m", "size": "n"})
            coarse = base.groupby(FALLBACK)[col].mean().rename("c")
            j = df[CELL].merge(fine, left_on=CELL, right_index=True, how="left")
            j = j.join(df[FALLBACK].merge(coarse, left_on=FALLBACK, right_index=True, how="left")["c"])
            # shrink thin cells towards the coarse (phase-level) mean
            k = 150.0
            n = j["n"].fillna(0)
            res[name] = ((j["m"].fillna(0) * n + j["c"].fillna(base[col].mean()) * k) / (n + k)).values
        return res

    bat = expect(df["faced"], {"E_bat_runs": "runs_bat", "E_bat_out": "batter_out"})
    bowl = expect(pd.Series(True, index=df.index), {"E_bowl_runs": "runs_bowler", "E_bowl_wkt": "bowler_wicket"})
    for k, v in {**bat, **bowl}.items():
        df[k] = v
    # ball-level components (lambda applied later so it can be varied cheaply)
    df["bat_rv"] = np.where(df["faced"], df["runs_bat"] - df["E_bat_runs"], 0.0)
    df["bat_wv"] = np.where(df["faced"], -(df["batter_out"].astype(float) - df["E_bat_out"]), 0.0)
    df["bowl_rv"] = df["E_bowl_runs"] - df["runs_bowler"]
    df["bowl_wv"] = df["bowler_wicket"].astype(float) - df["E_bowl_wkt"]
    return df


def _role_cols(role):
    if role == "bat":
        return dict(pid="batter", name="batter_name", rv="bat_rv", wv="bat_wv", mask="faced", k=config.SHRINK_BALLS_BAT,
                    min_balls=config.MIN_PRESSURE_BALLS_BAT)
    return dict(pid="bowler", name="bowler_name", rv="bowl_rv", wv="bowl_wv", mask=None, k=config.SHRINK_BALLS_BOWL,
                min_balls=config.MIN_PRESSURE_BALLS_BOWL)


def role_frame(df, role):
    c = _role_cols(role)
    d = df if c["mask"] is None else df[df[c["mask"]]]
    return d, c


def zscore(s):
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0


def player_ratings(df: pd.DataFrame, role: str, wicket_value=None, score_weights=None, shrink=None,
                   min_balls=None, min_matches=None, restrict_ids=None) -> pd.DataFrame:
    """df must already have high_pressure + expectation columns."""
    L = config.WICKET_VALUE_RUNS if wicket_value is None else wicket_value
    sw = score_weights or config.SCORE_WEIGHTS
    d, c = role_frame(df, role)
    k = shrink or c["k"]
    min_balls = min_balls or c["min_balls"]
    min_matches = config.MIN_PRESSURE_MATCHES if min_matches is None else min_matches
    d = d.assign(val=d[c["rv"]] + L * d[c["wv"]])
    pid = c["pid"]

    hp = d[d["high_pressure"]]
    lp = d[~d["high_pressure"]]
    agg_hp = hp.groupby(pid).agg(hp_balls=("val", "size"), hp_sum=("val", "sum"), hp_matches=("match_id", "nunique"))
    agg_lp = lp.groupby(pid).agg(lp_balls=("val", "size"), lp_sum=("val", "sum"))
    t = agg_hp.join(agg_lp, how="left").fillna({"lp_balls": 0, "lp_sum": 0})
    t["PV"] = 100 * t["hp_sum"] / (t["hp_balls"] + k)            # shrunk value / 100 pressure balls
    t["BV"] = 100 * t["lp_sum"] / (t["lp_balls"] + k)            # shrunk value / 100 normal balls
    t["resilience"] = t["PV"] - t["BV"]

    # consistency: share of matches (>= N pressure balls) with positive pressure value
    pm = hp.groupby([pid, "match_id"])["val"].agg(["sum", "size"]).reset_index()
    pm = pm[pm["size"] >= config.MIN_PRESSURE_BALLS_PER_MATCH]
    pm["pos"] = pm["sum"] > 0
    cons = pm.groupby(pid).agg(cons_matches=("pos", "size"), cons_pos=("pos", "sum"), match_sd=("sum", "std"))
    t = t.join(cons)
    pr = config.CONSIST_PRIOR
    t["consistency"] = (t["cons_pos"].fillna(0) + pr / 2) / (t["cons_matches"].fillna(0) + pr)

    # descriptive stats
    if role == "bat":
        for tag, g in (("hp", hp), ("lp", lp)):
            s = g.groupby(pid).agg(runs=("runs_bat", "sum"), balls=("faced", "sum"), outs=("batter_out", "sum"),
                                   bnd=("runs_bat", lambda x: (x >= 4).sum()), dots=("runs_bat", lambda x: (x == 0).sum()))
            t[f"{tag}_runs"], t[f"{tag}_outs"] = s["runs"], s["outs"]
            t[f"{tag}_SR"] = 100 * s["runs"] / s["balls"]
            t[f"{tag}_avg"] = s["runs"] / s["outs"].replace(0, np.nan)
            t[f"{tag}_bnd%"] = 100 * s["bnd"] / s["balls"]
    else:
        for tag, g in (("hp", hp), ("lp", lp)):
            s = g.groupby(pid).agg(runs=("runs_bowler", "sum"), balls=("legal", "sum"), wkts=("bowler_wicket", "sum"),
                                   dots=("runs_bowler", lambda x: (x == 0).sum()), n=("legal", "size"))
            t[f"{tag}_wkts"] = s["wkts"]
            t[f"{tag}_econ"] = 6 * s["runs"] / s["balls"]
            t[f"{tag}_dot%"] = 100 * s["dots"] / s["n"]
            t[f"{tag}_SR"] = s["balls"] / s["wkts"].replace(0, np.nan)   # balls per wicket

    # eligibility
    t["eligible"] = (t["hp_balls"] >= min_balls) & (t["hp_matches"] >= min_matches)
    if restrict_ids is not None:
        t["eligible"] &= t.index.isin(restrict_ids)
    e = t[t["eligible"]]
    t["z_value"] = zscore(e["PV"]).reindex(t.index)
    t["z_resil"] = zscore(e["resilience"]).reindex(t.index)
    t["z_cons"] = zscore(e["consistency"]).reindex(t.index)
    t["score"] = sw["value"] * t["z_value"] + sw["resilience"] * t["z_resil"] + sw["consistency"] * t["z_cons"]
    t = t.reset_index().rename(columns={pid: "player_id"})
    nm = d.drop_duplicates(pid).set_index(pid)[c["name"]]
    t.insert(1, "player", t["player_id"].map(nm))
    t["rank"] = t["score"].rank(ascending=False, method="min")
    return t.sort_values("score", ascending=False).reset_index(drop=True)


def split_half_reliability(df, role, n_rep=20, seed=7, wicket_value=None, shrink=None):
    """Spearman correlation of pressure value between two random halves of matches."""
    from scipy.stats import spearmanr
    L = config.WICKET_VALUE_RUNS if wicket_value is None else wicket_value
    d, c = role_frame(df, role)
    d = d[d["high_pressure"]].assign(val=lambda x: x[c["rv"]] + L * x[c["wv"]])
    mids = d["match_id"].unique()
    kk = shrink or c["k"]
    rng = np.random.default_rng(seed)
    cors = []
    for _ in range(n_rep):
        half = set(rng.choice(mids, len(mids) // 2, replace=False))
        a = d[d["match_id"].isin(half)].groupby(c["pid"])["val"].agg(["sum", "size"])
        b = d[~d["match_id"].isin(half)].groupby(c["pid"])["val"].agg(["sum", "size"])
        j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
        j = j[(j["size_a"] >= c["min_balls"] / 3) & (j["size_b"] >= c["min_balls"] / 3)]
        if len(j) < 15:
            continue
        pa = j["sum_a"] / (j["size_a"] + kk); pb = j["sum_b"] / (j["size_b"] + kk)
        cors.append(spearmanr(pa, pb)[0])
    return dict(mean=float(np.mean(cors)) if cors else np.nan, sd=float(np.std(cors)) if cors else np.nan, n=len(cors))


def player_meta(sup: dict) -> pd.DataFrame:
    """Latest team, and flags for IPL 2025/2026 squads and any 2026 activity."""
    m, comps = sup["mapping"], sup["competitions"]
    mm = m.merge(comps[["comp_id", "season"]], on="comp_id", how="left")
    last_date = sup["matches"].groupby("comp_id")["match_date"].max().rename("comp_last")
    mm = mm.merge(last_date, on="comp_id", how="left").sort_values("comp_last")
    latest = mm.drop_duplicates("player_id", keep="last").set_index("player_id")
    meta = pd.DataFrame({"latest_team": latest["team_name"], "last_seen": latest["comp_last"]})
    ipl = mm[mm["comp_name"].str.contains("IPL", na=False)]
    for yr in (2025, 2026):
        meta[f"IPL{yr}"] = meta.index.isin(ipl[ipl["comp_name"].str.contains(str(yr))]["player_id"])
    rr = mm[(mm["team_name"] == "Rajasthan Royals") & mm["comp_name"].str.contains("IPL 2026", na=False)]
    meta["RR_2026"] = meta.index.isin(rr["player_id"])
    meta["active_2026"] = meta.index.isin(mm[mm["season"].astype(str).str.contains("2026")]["player_id"])
    return meta


def batter_profile(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["faced"]]
    out = pd.DataFrame(index=d["batter"].unique())
    out.index.name = "player_id"
    if "batting_order" in d:
        o = d.groupby("batter")["batting_order"].median()
        out["usual_order"] = o
        out["role"] = pd.cut(o, [0, 2.5, 4.5, 6.5, 12], labels=["Opener", "Top-order", "Middle-order", "Lower-order"]).astype(str)
    out["death_share"] = d.groupby("batter")["phase"].apply(lambda s: (s == "Death").mean())
    out["pp_share"] = d.groupby("batter")["phase"].apply(lambda s: (s == "Powerplay").mean())
    if "batting_type" in d:
        out["hand"] = d.groupby("batter")["batting_type"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else "")
    if "pace_or_spin" in d:
        hp = d[d["high_pressure"]]
        for kind in ("Pace", "Spin"):
            g = hp[hp["pace_or_spin"] == kind].groupby("batter").agg(r=("runs_bat", "sum"), b=("faced", "sum"), o=("batter_out", "sum"))
            out[f"hp_SR_vs_{kind.lower()}"] = (100 * g["r"] / g["b"]).where(g["b"] >= 25)
            out[f"hp_balls_vs_{kind.lower()}"] = g["b"]
    return out


def bowler_profile(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df["bowler"].unique()); out.index.name = "player_id"
    ph = df.groupby(["bowler", "phase"]).size().unstack(fill_value=0)
    ph = ph.div(ph.sum(axis=1), axis=0)
    out["pp_share"] = ph.get("Powerplay"); out["mid_share"] = ph.get("Middle"); out["death_share"] = ph.get("Death")
    def lab(r):
        best = r[["pp_share", "mid_share", "death_share"]].astype(float).idxmax()
        top = r[best]
        return {"pp_share": "Powerplay", "mid_share": "Middle-overs", "death_share": "Death"}[best] + (" specialist" if top >= 0.45 else "-leaning")
    out["role"] = out.apply(lab, axis=1)
    if "bowling_type" in df:
        out["style"] = df.groupby("bowler")["bowling_type"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else "")
    if "pace_or_spin" in df:
        out["kind"] = df.groupby("bowler")["pace_or_spin"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else "")
    return out


def fit_text(r: pd.Series, role: str) -> str:
    """One-line 'how he strengthens RR' blurb built purely from the data."""
    if role == "bat":
        bits = [f"{r.get('role', 'Batter')} ({r.get('hand', '')})".replace(" ()", "")]
        bits.append(f"strike rate {r['hp_SR']:.0f} under pressure vs {r['lp_SR']:.0f} otherwise")
        if r.get("death_share", 0) >= 0.3:
            bits.append(f"{r['death_share']:.0%} of balls in the death overs")
        sp, pc = r.get("hp_SR_vs_spin"), r.get("hp_SR_vs_pace")
        if pd.notna(sp) and pd.notna(pc) and abs(sp - pc) > 15:
            bits.append("stronger against " + ("spin" if sp > pc else "pace") + f" ({max(sp, pc):.0f} vs {min(sp, pc):.0f} SR)")
        return "; ".join(bits) + "."
    bits = [f"{r.get('role', 'Bowler')}" + (f", {r['style']}" if isinstance(r.get("style"), str) and r.get("style") else "")]
    bits.append(f"economy {r['hp_econ']:.1f} under pressure vs {r['lp_econ']:.1f} otherwise")
    if pd.notna(r.get("hp_SR")):
        bits.append(f"a wicket every {r['hp_SR']:.0f} pressure balls")
    return "; ".join(bits) + "."
