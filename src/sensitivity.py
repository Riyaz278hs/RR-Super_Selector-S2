"""How stable is the top-10 when modelling choices change?"""
from __future__ import annotations
import itertools
import pandas as pd
import config
from . import metrics
from . import pressure


def top10_stability(df, role, n=10, base_pctl=None):
    base_pctl = base_pctl or config.HIGH_PRESSURE_PCTL
    d0, _ = pressure.label_pressure(df, base_pctl)
    base = metrics.player_ratings(d0, role)
    base_top = list(base[base.eligible].head(n)["player_id"])
    rows = []
    for pctl, L, sw_name, sw in itertools.product(
            (60, base_pctl, 80), (10, config.WICKET_VALUE_RUNS, 30),
            ("default", "value-only"), (config.SCORE_WEIGHTS, dict(value=1, resilience=0, consistency=0))):
        pass
    scen = []
    for pctl in (60, base_pctl, 80):
        scen.append((f"pressure pctl={pctl}", dict(pctl=pctl)))
    for L in (10, 30):
        scen.append((f"wicket value={L}", dict(L=L)))
    scen.append(("value only (no resilience/consistency)", dict(sw=dict(value=1, resilience=0, consistency=0))))
    scen.append(("consistency-heavy", dict(sw=dict(value=0.3, resilience=0.2, consistency=0.5))))
    scen.append(("stronger shrinkage (2x)", dict(shrink=2)))
    for name, kw in scen:
        d = df
        if "pctl" in kw:
            d, _ = pressure.label_pressure(df, kw["pctl"])
        k = None
        if "shrink" in kw:
            k = (config.SHRINK_BALLS_BAT if role == "bat" else config.SHRINK_BALLS_BOWL) * kw["shrink"]
        t = metrics.player_ratings(d, role, wicket_value=kw.get("L"), score_weights=kw.get("sw"), shrink=k)
        top = list(t[t.eligible].head(n)["player_id"])
        rows.append(dict(scenario=name, overlap_with_base=len(set(top) & set(base_top)),
                         rank1=t.iloc[0]["player"], top10_ids=top))
    out = pd.DataFrame(rows)
    # how often each base-top-10 player stays in top 10
    freq = pd.Series({pid: sum(pid in r for r in out["top10_ids"]) for pid in base_top}) / len(out)
    return out.drop(columns="top10_ids"), freq
