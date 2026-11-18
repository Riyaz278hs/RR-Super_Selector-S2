"""Builds the Methodology Report (.docx) from pipeline results so text, tables and rankings always agree."""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib

from . import pressure
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Pt, Inches
import config
from . import metrics

PINK, NAVY = "#EA1B85", "#0B1F4B"


def _charts(df, tabs):
    out = config.OUT_DIR
    # 1 PI by over
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    for inn, c in ((1, NAVY), (2, PINK)):
        s = df[df.innings == inn].groupby("over_idx")["PI"].mean()
        ax.plot(s.index + 1, s.values, marker="o", ms=3, color=c, label=f"Innings {inn}")
    ax.set_xlabel("Over"); ax.set_ylabel("Mean Pressure Index"); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "fig_pi_by_over.png", dpi=170); plt.close(fig)
    # 2 outcome by PI decile
    v = pressure.validate_pressure(df)
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.bar(v["decile"], 100 * v["wicket_rate"], color=NAVY)
    ax.set_xlabel("Pressure Index decile (1 = calmest)"); ax.set_ylabel("Wicket rate (% of balls)")
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(out / "fig_wkt_by_decile.png", dpi=170); plt.close(fig)
    # 3 top-10 bars
    paths = []
    for role, title in (("bat", "Batters"), ("bowl", "Bowlers")):
        t = tabs[role]; t = t[t.eligible].head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        ax.barh(t["player"], t["score"], color=PINK)
        ax.set_xlabel("Composite pressure score (z-units)"); ax.set_title(f"Top 10 {title} under pressure", loc="left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
        p = out / f"fig_top10_{role}.png"; fig.savefig(p, dpi=170); plt.close(fig); paths.append(p)
    return paths


def _table(doc, frame, widths=None):
    t = doc.add_table(rows=1, cols=len(frame.columns)); t.style = "Light Grid Accent 1"
    for i, c in enumerate(frame.columns):
        t.rows[0].cells[i].text = str(c)
    for r in frame.itertuples(index=False):
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(8.5)
    doc.add_paragraph()


def build_report(df, sup, info, tabs, rel, sens, freq):
    figs = _charts(df, tabs)
    thr = info["threshold"]
    w = config.PI_WEIGHTS
    hp = df["high_pressure"].mean()
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"; doc.styles["Normal"].font.size = Pt(10.5)
    doc.add_heading("Identifying the Most Reliable Players Under Pressure", 0)
    doc.add_paragraph("SupeRR Selector Season 2 — Round 1 methodology report")

    n_ipl = df[df.tournament == "IPL"]["match_id"].nunique(); n_t20i = df[df.tournament == "T20I"]["match_id"].nunique()
    doc.add_heading("1. Summary", 1)
    doc.add_paragraph(
        f"We built a ball-level Pressure Index (PI, 0–100) from match state, judged every batter and bowler against what an average player "
        f"produces in the same context, and ranked players on how much value they add in high-pressure balls (PV), how well they hold up "
        f"relative to their own normal level (resilience) and how often they deliver match after match (consistency). "
        f"The analysis covers {info['n_matches']:,} matches ({n_ipl} IPL, {n_t20i} T20I) and {info['n_balls']:,} deliveries from 2022 onwards.")

    doc.add_heading("Data handling", 2)
    for n_ in info["notes"]:
        doc.add_paragraph(n_, style="List Bullet")
    doc.add_paragraph("Players are ranked only if they appear in a 2026 squad (IPL 2026 or a 2026 T20I series), because the output is meant to guide IPL 2027 auction decisions." if config.REQUIRE_ACTIVE_2026 else "All players meeting the sample thresholds are ranked.", style="List Bullet")

    doc.add_heading("2. Pressure definition", 1)
    doc.add_paragraph(
        "Pressure is a property of the match situation before each ball and is identical for the batter and the bowler. "
        "Four components, each scaled to 0–1, are combined:")
    for s in [
        f"Stage ({w['stage']:.0%}): (balls bowled / 120)². Convex, so the last overs weigh far more than the first.",
        f"Closeness ({w['close']:.0%}): 1 − |deficit| / {config.CLOSE_SCALE:.0f}. Deficit is runs behind par pace (1st innings: par score at that ball − current score; "
        "2nd innings: runs needed − par runs still to come). The contest is tightest when deficit ≈ 0.",
        f"Demand ({w['demand']:.0%}): max(deficit, 0) / {config.DEMAND_SCALE:.0f}. The batting side is behind par and must take risk.",
        f"Wickets ({w['wkt']:.0%}): wickets lost / {config.WKT_SATURATION}, capped at 1. A depleted batting side is under stress, and the bowling side is on top.",
    ]:
        doc.add_paragraph(s, style="List Bullet")
    doc.add_paragraph(
        "Par curves are measured from the data itself (mean first-innings score and runs-remaining at every ball, separately for IPL and T20I), "
        "so no outside assumptions are needed. In chases with RRR > 24 and fewer than 4 overs left the match is effectively lost and PI is damped by 40%.")
    doc.add_paragraph(f"A ball is high-pressure if PI ≥ {thr:.1f}, the {config.HIGH_PRESSURE_PCTL}th percentile of all balls ({hp:.0%} of balls).")
    doc.add_picture(str(config.OUT_DIR / "fig_pi_by_over.png"), width=Inches(5.8))
    doc.add_paragraph("Figure 1. Mean PI rises through an innings as stage, wickets and the target start to bite.")
    doc.add_picture(str(config.OUT_DIR / "fig_wkt_by_decile.png"), width=Inches(5.8))
    v_ = pressure.validate_pressure(df)
    doc.add_paragraph(f"Figure 2. Validation: the wicket rate climbs from {100 * v_['wicket_rate'].iloc[0]:.1f}% of balls in the calmest decile to {100 * v_['wicket_rate'].iloc[-1]:.1f}% in the tensest, "
                      f"and the dot-ball rate falls from {100 * v_['dot_rate'].iloc[0]:.0f}% to {100 * v_['dot_rate'].iloc[-1]:.0f}% as batters are forced to attack. The index separates tense moments from ordinary ones.")

    doc.add_heading("3. Performance measurement", 1)
    doc.add_paragraph(
        "For each delivery we estimate the average outcome in the same context (cell = tournament × innings × over × wickets-down bucket, thin cells "
        "shrunk to the phase-level mean). Value per ball:")
    doc.add_paragraph(f"Batter = (runs − E[runs]) − {config.WICKET_VALUE_RUNS:.0f}·(out − E[out])", style="List Bullet")
    doc.add_paragraph(f"Bowler = (E[runs] − runs conceded) + {config.WICKET_VALUE_RUNS:.0f}·(wicket − E[wicket])  (run-outs are not credited to the bowler; byes/leg-byes are not charged)", style="List Bullet")
    doc.add_paragraph(
        f"PV is the summed value over a player's high-pressure balls per 100 balls, shrunk toward zero with {config.SHRINK_BALLS_BAT} (bat) / "
        f"{config.SHRINK_BALLS_BOWL} (bowl) pseudo-balls of average play, so small samples cannot top the list. "
        "Baseline BV is the same quantity on the player's normal-pressure balls; Resilience = PV − BV (how a player's usual level survives pressure). "
        f"Consistency is the share of matches (≥{config.MIN_PRESSURE_BALLS_PER_MATCH} pressure balls) with positive pressure value, with a Beta-style prior of {config.CONSIST_PRIOR} pseudo-matches.")
    lam = {L: metrics.split_half_reliability(df, "bat", n_rep=15, wicket_value=L)["mean"] for L in (5, 10, 20)}
    lamb = {L: metrics.split_half_reliability(df, "bowl", n_rep=15, wicket_value=L)["mean"] for L in (5, 10, 20)}
    doc.add_paragraph(
        "Why a wicket is worth " + f"{config.WICKET_VALUE_RUNS:.0f} runs: wickets are rare, so heavy wicket weights make rankings noisy. Split-half reliability (Spearman ρ, batters / bowlers) "
        + ", ".join(f"at {k} runs: {lam[k]:.2f} / {lamb[k]:.2f}" for k in lam) + f". We use {config.WICKET_VALUE_RUNS:.0f} as a balance between stability and respecting that a dismissal is costly; the sensitivity tables show the effect of 10 vs 30.")
    sw = config.SCORE_WEIGHTS
    doc.add_paragraph(f"Final score = {sw['value']:.0%}·z(PV) + {sw['resilience']:.0%}·z(Resilience) + {sw['consistency']:.0%}·z(Consistency), z-scored across eligible players.")

    doc.add_heading("4. Fair comparisons", 1)
    for s in [
        f"Sample size: eligibility needs ≥{config.MIN_PRESSURE_BALLS_BAT} high-pressure balls faced (batters) or ≥{config.MIN_PRESSURE_BALLS_BOWL} bowled (bowlers) and ≥{config.MIN_PRESSURE_MATCHES} matches; plus shrinkage as above.",
        "Context: expectations are computed within tournament, innings, over and wickets, so death-over hitters, powerplay bowlers and chasers are compared with their true peers.",
        "IPL vs T20I: both used. Each is benchmarked against its own league average and its own par curves, then values are pooled. T20I opposition quality varies; we do not adjust for it (see limitations).",
        "Rain-affected matches (revised overs or target) are excluded because the 20-over pressure maths does not apply.",
    ]:
        doc.add_paragraph(s, style="List Bullet")

    doc.add_heading("5. Results and evidence", 1)
    for role, title, fig in (("bat", "Top 10 batters", figs[0]), ("bowl", "Top 10 bowlers", figs[1])):
        doc.add_heading(title, 2)
        t = tabs[role]; t = t[t.eligible].head(10)
        if role == "bat":
            f = t[["rank", "player", "hp_balls", "hp_SR", "lp_SR", "hp_avg", "PV", "resilience", "consistency", "score"]].copy()
            f.columns = ["#", "Player", "HP balls", "HP SR", "Normal SR", "HP avg", "PV", "Resil.", "Consist.", "Score"]
        else:
            f = t[["rank", "player", "hp_balls", "hp_econ", "lp_econ", "hp_wkts", "PV", "resilience", "consistency", "score"]].copy()
            f.columns = ["#", "Player", "HP balls", "HP econ", "Normal econ", "HP wkts", "PV", "Resil.", "Consist.", "Score"]
        f["#"] = f["#"].astype(int)
        _table(doc, f.round(2))
        doc.add_picture(str(fig), width=Inches(5.6))
        doc.add_paragraph("How each player could strengthen the Royals", style="Heading 3")
        for r_ in t.itertuples():
            doc.add_paragraph(f"{int(r_.rank)}. {r_.player} ({r_.latest_team}): {r_.fit}", style="List Bullet")

    doc.add_heading("6. Reliability and limitations", 1)
    doc.add_paragraph(
        f"Split-half check: randomly splitting matches in two and correlating players' pressure value gives Spearman ρ = {rel['bat']['mean']:.2f} (batters) "
        f"and {rel['bowl']['mean']:.2f} (bowlers) averaged over {rel['bat']['n']} splits. Pressure samples are small, so rankings are noisy by nature; this is why we shrink, require consistency and test stability.")
    for role, name in (("bat", "batters"), ("bowl", "bowlers")):
        s = sens[role].copy(); s.columns = ["Scenario", "Top-10 overlap with base (of 10)", "Rank-1 player"]
        doc.add_paragraph(f"Sensitivity of the top-10 {name} to modelling choices:")
        _table(doc, s)
        doc.add_paragraph("Share of scenarios in which each base-case top-10 player stays in the top 10: " +
                          "; ".join(f"{tabs[role].set_index('player_id').loc[i, 'player']} {v:.0%}" for i, v in freq[role].items()))
    for s in [f"Wicket value ({config.WICKET_VALUE_RUNS:.0f} runs) and the PI weights are judgement calls; sensitivity tables show their impact.",
              "No adjustment for opposition strength, venue or pitch; a player facing weak attacks in T20Is may look better.",
              "Batters' value is not split by batting position/role (finisher vs anchor).",
              "Form and age are not modelled; a recency or availability screen is advisable before auction decisions.",
              "Pressure is inferred from scoreboard state only; it ignores unrecorded factors (tournament stage, crowd, match importance). Knockout flag is available in the data for a future extension."]:
        doc.add_paragraph(s, style="List Bullet")

    doc.add_heading("7. Reproducibility", 1)
    doc.add_paragraph("Code: Python (pandas, numpy, scipy). Run `python run_pipeline.py --report`; every parameter is in config.py and the Streamlit dashboard (`streamlit run app.py`) recomputes the rankings interactively.")
    p = config.OUT_DIR / "Methodology_Report.docx"
    doc.save(p)
    return p
