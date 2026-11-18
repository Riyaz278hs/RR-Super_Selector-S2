"""Streamlit dashboard — run with:  streamlit run app.py"""
from __future__ import annotations
import io, os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from src import loader, pressure, metrics, pipeline, sensitivity

PINK, NAVY, GREY, LIGHT = "#EA1B85", "#0B1F4B", "#8A93A8", "#F4F5FA"
st.set_page_config(page_title="RR · Reliable Players Under Pressure", page_icon="🏏", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(f"""<style>
.block-container {{ padding-top: 1rem; max-width: 1400px; }}
.hero {{ background: linear-gradient(120deg,{NAVY} 0%, #1B2F6B 60%, {PINK} 140%); color:#fff; padding: 1.4rem 1.8rem;
        border-radius: 14px; margin-bottom: 1rem; }}
.hero h1 {{ color:#fff; font-size: 1.9rem; margin:0; font-weight:800; letter-spacing:-.5px; }}
.hero p {{ color:#DCE1F2; margin:.25rem 0 0; font-size:.95rem; }}
.hero .tag {{ display:inline-block; background:{PINK}; color:#fff; border-radius:999px; padding:2px 12px; font-size:.72rem;
        font-weight:700; letter-spacing:.8px; margin-bottom:.5rem; }}
.kpi {{ background:{LIGHT}; border-radius:12px; padding:.9rem 1.1rem; border-left:5px solid {PINK}; }}
.kpi .v {{ font-size:1.7rem; font-weight:800; color:{NAVY}; line-height:1.1; }}
.kpi .l {{ font-size:.78rem; color:{GREY}; text-transform:uppercase; letter-spacing:.6px; }}
.pick {{ background:#fff; border:1px solid #E3E6F0; border-radius:12px; padding:.6rem .9rem; margin-bottom:.45rem; }}
.pick .n {{ display:inline-block; width:1.9rem; height:1.9rem; line-height:1.9rem; text-align:center; background:{NAVY}; color:#fff;
        border-radius:50%; font-weight:700; margin-right:.6rem; }}
.pick .nm {{ font-weight:700; color:{NAVY}; }} .pick .sub {{ color:{GREY}; font-size:.8rem; }}
.pick .fit {{ color:#39425E; font-size:.82rem; margin-top:.2rem; }}
.insight {{ background:#FFF3F9; border-left:4px solid {PINK}; padding:.6rem .9rem; border-radius:8px; margin:.35rem 0; color:{NAVY}; }}
h2,h3 {{ color:{NAVY}; }}
button[data-baseweb="tab"] {{ font-weight:600; }}
</style>""", unsafe_allow_html=True)


def style(fig, h=None):
    fig.update_layout(font=dict(family="Inter, Segoe UI, sans-serif", color=NAVY), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=50, b=10), colorway=[PINK, NAVY, GREY, "#F5A3CE", "#5B6FA8"],
                      title_font=dict(size=15), legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(showgrid=False, linecolor="#D5D9E6"); fig.update_yaxes(gridcolor="#ECEEF5", zeroline=False)
    if h: fig.update_layout(height=h)
    return fig


def kpi(col, value, label):
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner=False)
def get_sup():
    return loader.load_supporting()


@st.cache_data(show_spinner=False)
def header_cols(src, up):
    return list(pd.read_csv(io.BytesIO(up) if up else src, nrows=0).columns)


@st.cache_data(show_spinner="Building ball table & pressure index (first run only, ~20 s)…")
def build(src, mtime, up, schema_items, w_items, close_scale, demand_scale, wkt_sat):
    raw = pd.read_csv(io.BytesIO(up) if up else src, low_memory=False)
    df, info = loader.normalise_bbb(raw, dict(schema_items), get_sup())
    df = pressure.add_pressure(df, weights=dict(w_items), close_scale=close_scale, demand_scale=demand_scale, wkt_sat=wkt_sat)
    df = metrics.add_expectations(df)
    return df, info


@st.cache_data(show_spinner=False)
def meta_tbl():
    return metrics.player_meta(get_sup())


@st.cache_data(show_spinner="Rating players…")
def ratings(_df, key, role, **kw):
    t = metrics.player_ratings(_df, role, **kw)
    t = t.merge(meta_tbl(), left_on="player_id", right_index=True, how="left")
    prof = metrics.batter_profile(_df) if role == "bat" else metrics.bowler_profile(_df)
    t = t.merge(prof, left_on="player_id", right_index=True, how="left")
    t["fit"] = t.apply(lambda r: metrics.fit_text(r, role) if r["eligible"] else "", axis=1)
    return t


try:
    sup = get_sup()
except Exception as e:
    st.error(f"Could not load the four supporting CSVs from /data: {e}"); st.stop()

sb = st.sidebar
sb.markdown(f"### 🏏 Controls")
found = loader.find_bbb_file()
up = sb.file_uploader("Ball-by-ball CSV (optional override)", type="csv")
upb = up.getvalue() if up is not None else None
src = str(found) if found else ""
if not upb and not src:
    st.info("Put your ball-by-ball CSV in the /data folder, or upload it in the sidebar."); st.stop()
if not upb and "sample_" in src:
    st.warning("Using SIMULATED sample data. Rankings are not real.")
sb.caption(f"Data: {'uploaded file' if upb else Path(src).name}")
cols_ = header_cols(src, upb)
guess = loader.detect_schema(cols_)
schema = {}
with sb.expander("Column mapping (auto-detected)"):
    opts = ["(none)"] + cols_
    for k in loader.ALIASES:
        v = st.selectbox(k, opts, index=opts.index(guess[k]) if guess.get(k) in opts else 0, key=f"map_{k}")
        schema[k] = None if v == "(none)" else v

sb.markdown("**Pressure definition**")
pctl = sb.slider("High-pressure = top X% of balls", 50, 95, config.HIGH_PRESSURE_PCTL, 1, help="Percentile cut-off on the Pressure Index")
with sb.expander("Index weights & scales"):
    w_close = st.slider("Closeness", 0.0, 1.0, config.PI_WEIGHTS["close"], 0.05)
    w_stage = st.slider("Stage of innings", 0.0, 1.0, config.PI_WEIGHTS["stage"], 0.05)
    w_wkt = st.slider("Wickets lost", 0.0, 1.0, config.PI_WEIGHTS["wkt"], 0.05)
    w_dem = st.slider("Demand (behind par)", 0.0, 1.0, config.PI_WEIGHTS["demand"], 0.05)
    close_scale = st.slider("Closeness scale (runs)", 10, 60, int(config.CLOSE_SCALE))
    demand_scale = st.slider("Demand scale (runs)", 10, 60, int(config.DEMAND_SCALE))
    wkt_sat = st.slider("Wickets at saturation", 4, 10, config.WKT_SATURATION)
tot = max(w_close + w_stage + w_wkt + w_dem, 1e-9)
weights = dict(close=w_close / tot, stage=w_stage / tot, wkt=w_wkt / tot, demand=w_dem / tot)

sb.markdown("**Rating model**")
L = sb.slider("Value of a wicket (runs)", 5, 35, int(config.WICKET_VALUE_RUNS))
with sb.expander("Eligibility, shrinkage & score weights"):
    shrink_mult = st.slider("Shrinkage strength (× default)", 0.5, 3.0, 1.0, 0.25)
    min_b_bat = st.number_input("Min pressure balls · batters", 30, 500, config.MIN_PRESSURE_BALLS_BAT, 10)
    min_b_bowl = st.number_input("Min pressure balls · bowlers", 30, 500, config.MIN_PRESSURE_BALLS_BOWL, 10)
    min_m = st.number_input("Min pressure matches", 3, 40, config.MIN_PRESSURE_MATCHES)
    s_v = st.slider("Weight · pressure value", 0.0, 1.0, config.SCORE_WEIGHTS["value"], 0.05)
    s_r = st.slider("Weight · resilience", 0.0, 1.0, config.SCORE_WEIGHTS["resilience"], 0.05)
    s_c = st.slider("Weight · consistency", 0.0, 1.0, config.SCORE_WEIGHTS["consistency"], 0.05)
st_ = max(s_v + s_r + s_c, 1e-9)
score_w = dict(value=s_v / st_, resilience=s_r / st_, consistency=s_c / st_)

sb.markdown("**Filters**")
try:
    mtime = os.path.getmtime(src) if (src and not upb) else 0
    df_all, info = build(src, mtime, upb, tuple(sorted(schema.items(), key=lambda x: x[0])), tuple(weights.items()),
                         float(close_scale), float(demand_scale), int(wkt_sat))
except Exception as e:
    st.error(f"Could not process the file: {e}\n\nCheck the column mapping in the sidebar."); st.stop()
df_all, thr = pressure.label_pressure(df_all, pctl)
df_all["year"] = pd.to_datetime(df_all["match_date"]).dt.year
tour = sb.multiselect("Tournament", ["IPL", "T20I"], default=["IPL", "T20I"])
years = sorted(df_all["year"].unique())
yr = sb.multiselect("Match year", years, default=years)
df = df_all[df_all["tournament"].isin(tour) & df_all["year"].isin(yr)]
meta = meta_tbl()
only_active = sb.checkbox("Only players in a 2026 squad (auction-relevant)", value=config.REQUIRE_ACTIVE_2026)
only_ipl = sb.checkbox("Only players in an IPL 2025/26 squad", value=False)
restrict = None
if only_active or only_ipl:
    m_ = meta
    if only_active: m_ = m_[m_["active_2026"]]
    if only_ipl: m_ = m_[m_["IPL2025"] | m_["IPL2026"]]
    restrict = tuple(sorted(m_.index))
for n in info["notes"]:
    sb.caption("ℹ️ " + n)

key = (tuple(tour), tuple(yr), pctl, tuple(weights.values()), close_scale, demand_scale, wkt_sat, L, shrink_mult,
       min_b_bat, min_b_bowl, min_m, tuple(score_w.values()), only_active, only_ipl, src, upb is not None)
R = {}
for role, mb, k0 in (("bat", min_b_bat, config.SHRINK_BALLS_BAT), ("bowl", min_b_bowl, config.SHRINK_BALLS_BOWL)):
    R[role] = ratings(df, key, role, wicket_value=L, score_weights=score_w, shrink=k0 * shrink_mult, min_balls=int(mb),
                      min_matches=int(min_m), restrict_ids=set(restrict) if restrict else None)

# ------------------------------------------------------------------ header
st.markdown(f"""<div class="hero"><span class="tag">SUPERR SELECTOR · SEASON 2 · ROUND 1</span>
<h1>Most Reliable Players Under Pressure</h1>
<p>Who delivers when the game is on the line? Ball-by-ball Pressure Index across IPL and T20Is (2022 → 2026), turned into batting and bowling rankings for the IPL 2027 auction.</p></div>""",
            unsafe_allow_html=True)

T = st.tabs(["📊 Executive summary", "🧪 Pressure model", "🏏 Batters", "🎯 Bowlers", "🔍 Player explorer",
             "🧩 Squad fit & auction", "✅ Reliability", "📤 Submission"])


def top_cards(role, n=10, with_fit=True):
    t = R[role]; e = t[t.eligible].head(n)
    for _, r in e.iterrows():
        sub = (f"{r['latest_team']} · {int(r['hp_balls'])} pressure balls · PV {r['PV']:.0f} · consistency {r['consistency']:.0%}")
        fit = f'<div class="fit">{r["fit"]}</div>' if with_fit and r.get("fit") else ""
        st.markdown(f'<div class="pick"><span class="n">{int(r["rank"])}</span><span class="nm">{r["player"]}</span>'
                    f'<div class="sub">{sub}</div>{fit}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ 1 executive summary
with T[0]:
    c = st.columns(5)
    kpi(c[0], f"{df['match_id'].nunique():,}", "Matches analysed")
    kpi(c[1], f"{len(df):,}", "Deliveries")
    kpi(c[2], f"{df['high_pressure'].mean():.0%}", "High-pressure balls")
    kpi(c[3], f"{thr:.0f}", "Pressure Index cut-off")
    kpi(c[4], f"{int(R['bat'].eligible.sum())} / {int(R['bowl'].eligible.sum())}", "Eligible batters / bowlers")
    st.write("")
    hp, lp = df[df.high_pressure & df.legal], df[~df.high_pressure & df.legal]
    ins = []
    ins.append(f"Under high pressure, batters score <b>{hp['runs_total'].mean() * 6:.1f}</b> runs/over and lose a wicket every "
               f"<b>{1 / max(hp['is_wicket'].mean(), 1e-9):.0f}</b> balls, versus <b>{lp['runs_total'].mean() * 6:.1f}</b> and "
               f"<b>{1 / max(lp['is_wicket'].mean(), 1e-9):.0f}</b> in ordinary situations.")
    for role, nm in (("bat", "batter"), ("bowl", "bowler")):
        e = R[role][R[role].eligible]
        if len(e):
            r = e.sort_values("resilience", ascending=False).iloc[0]
            ins.append(f"Biggest riser among {nm}s: <b>{r['player']}</b> (+{r['resilience']:.0f} value/100 balls under pressure vs own normal level).")
            c2 = e[e.hp_matches >= 15].sort_values("consistency", ascending=False)
            if len(c2):
                r = c2.iloc[0]
                ins.append(f"Most consistent {nm} (≥15 pressure matches): <b>{r['player']}</b>, positive in {r['consistency']:.0%} of matches.")
    for t_ in ins:
        st.markdown(f'<div class="insight">{t_}</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    a.subheader("Top 10 batters"); b.subheader("Top 10 bowlers")
    with a: top_cards("bat")
    with b: top_cards("bowl")

# ------------------------------------------------------------------ 2 pressure model
with T[1]:
    st.markdown(f"""**Pressure Index (0–100)** is computed from the match state *before* each ball and is shared by batter and bowler:

`PI = 100 × ({weights['close']:.2f}·Closeness + {weights['stage']:.2f}·Stage + {weights['wkt']:.2f}·Wickets + {weights['demand']:.2f}·Demand)`

**Stage** = (balls bowled / 120)² · **Wickets** = lost / {wkt_sat} · **Deficit** = runs behind par pace (par measured from the data) ·
**Closeness** = 1 − |deficit| / {close_scale} · **Demand** = max(deficit, 0) / {demand_scale}.  A ball is *high-pressure* if PI ≥ **{thr:.1f}** (top {100 - pctl}%).""")
    v = pressure.validate_pressure(df)
    c1, c2 = st.columns(2)
    f = go.Figure()
    f.add_bar(x=v["decile"], y=100 * v["wicket_rate"], name="Wicket %", marker_color=NAVY)
    f.add_scatter(x=v["decile"], y=100 * v["dot_rate"], name="Dot-ball %", yaxis="y2", line=dict(color=PINK, width=3))
    f.update_layout(title="Outcomes by Pressure Index decile", xaxis_title="PI decile (1 = calm, 10 = tense)", yaxis_title="Wicket %",
                    yaxis2=dict(overlaying="y", side="right", title="Dot %", showgrid=False))
    c1.plotly_chart(style(f), use_container_width=True)
    g = df.groupby(["innings", "over_idx"])["PI"].mean().reset_index()
    g["over"] = g["over_idx"] + 1; g["innings"] = "Innings " + g["innings"].astype(str)
    c2.plotly_chart(style(px.line(g, x="over", y="PI", color="innings", markers=True, title="Mean pressure by over",
                                  color_discrete_sequence=[NAVY, PINK])), use_container_width=True)
    hm = df.assign(w=df["wkts_before"].clip(upper=8)).groupby(["w", "over_idx"])["PI"].mean().unstack()
    fig = go.Figure(go.Heatmap(z=hm.values, x=hm.columns + 1, y=hm.index, colorscale=[[0, "#FFFFFF"], [1, PINK]], colorbar=dict(title="PI")))
    fig.update_layout(title="Mean PI by over and wickets lost", xaxis_title="Over", yaxis_title="Wickets lost")
    st.plotly_chart(style(fig), use_container_width=True)
    c3, c4 = st.columns(2)
    c3.plotly_chart(style(px.histogram(df.sample(min(len(df), 60000), random_state=1), x="PI", color="phase", nbins=60,
                                       title="PI distribution by phase", barmode="overlay", opacity=.7,
                                       color_discrete_sequence=[NAVY, GREY, PINK])), use_container_width=True)
    c4.dataframe(v.round(3), use_container_width=True, hide_index=True)


def role_tab(role, label):
    t = R[role]; e = t[t.eligible].copy()
    if e.empty:
        st.warning("No players pass the eligibility thresholds; lower them in the sidebar."); return
    e["rank"] = e["rank"].astype(int)
    extra = ["hp_SR", "lp_SR", "hp_avg", "hp_bnd%"] if role == "bat" else ["hp_econ", "lp_econ", "hp_wkts", "hp_dot%"]
    cols = ["rank", "player", "latest_team", "hp_balls", "hp_matches", "PV", "BV", "resilience", "consistency", "score"] + extra
    n = st.slider(f"Rows ({label})", 10, 60, 25, key=f"n_{role}")
    st.dataframe(e[cols].head(n).round(2), use_container_width=True, hide_index=True,
                 column_config={"score": st.column_config.ProgressColumn("score", min_value=float(e.score.min()), max_value=float(e.score.max()), format="%.2f"),
                                "consistency": st.column_config.ProgressColumn("consistency", min_value=0.0, max_value=1.0, format="%.2f")})
    st.download_button("Download full ranking (CSV)", t.to_csv(index=False), f"ratings_{role}.csv", key=f"dl_{role}")
    top = e.head(10)
    fig = px.scatter(e, x="PV", y="consistency", size="hp_balls", hover_name="player", color_discrete_sequence=["#C9CEDD"],
                     labels={"PV": "Pressure value per 100 balls", "consistency": "Consistency"},
                     title=f"{label}: value vs consistency (top 10 highlighted)")
    fig.add_scatter(x=top["PV"], y=top["consistency"], mode="markers+text", text=top["player"], textposition="top center",
                    marker=dict(color=PINK, size=12), name="Top 10")
    st.plotly_chart(style(fig, 460), use_container_width=True)
    lo, hi = float(min(e["BV"].min(), e["PV"].min())), float(max(e["BV"].max(), e["PV"].max()))
    fig2 = px.scatter(e, x="BV", y="PV", hover_name="player", color="resilience", color_continuous_scale=["#8A93A8", "#FFFFFF", PINK],
                      title="Normal-pressure level vs pressure level (above the line = rises under pressure)",
                      labels={"BV": "Value /100 balls, normal situations", "PV": "Value /100 balls, high pressure"})
    fig2.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dot", color=GREY))
    st.plotly_chart(style(fig2, 460), use_container_width=True)


with T[2]: role_tab("bat", "Batters")
with T[3]: role_tab("bowl", "Bowlers")

# ------------------------------------------------------------------ 5 player explorer
with T[4]:
    role = st.radio("Role", ["bat", "bowl"], horizontal=True, format_func=lambda r: "Batter" if r == "bat" else "Bowler")
    t = R[role]
    names = t.sort_values("hp_balls", ascending=False)["player"].dropna().tolist()
    pick = st.selectbox("Player (sorted by pressure balls)", names)
    row = t[t["player"] == pick].iloc[0]
    d, c = metrics.role_frame(df, role)
    d = d[d[c["pid"]] == row["player_id"]].assign(val=lambda x: x[c["rv"]] + L * x[c["wv"]])
    k = st.columns(5)
    kpi(k[0], int(row["rank"]) if row["eligible"] and pd.notna(row["rank"]) else "n/a", "Rank (eligible only)")
    kpi(k[1], f"{row['PV']:.1f}", "Pressure value /100")
    kpi(k[2], f"{row['BV']:.1f}", "Normal value /100")
    kpi(k[3], f"{row['consistency']:.0%}", "Consistency")
    kpi(k[4], int(row["hp_balls"]), "Pressure balls")
    if row.get("fit"):
        st.markdown(f'<div class="insight">{row["fit"]}</div>', unsafe_allow_html=True)
    pm = d[d["high_pressure"]].groupby(["match_id", "match_date"])["val"].agg(["sum", "size"]).reset_index()
    pm = pm[pm["size"] >= config.MIN_PRESSURE_BALLS_PER_MATCH].sort_values("match_date")
    if len(pm):
        pm["colour"] = np.where(pm["sum"] > 0, "Above expected", "Below expected")
        st.plotly_chart(style(px.bar(pm, x="match_date", y="sum", color="colour", title="Pressure value in each match",
                                     color_discrete_map={"Above expected": PINK, "Below expected": "#B9BFD0"})), use_container_width=True)
    a, b = st.columns(2)
    ph = d.groupby(["phase", "high_pressure"])["val"].mean().reset_index()
    ph["val"] *= 100; ph["Situation"] = np.where(ph["high_pressure"], "High pressure", "Normal")
    a.plotly_chart(style(px.bar(ph, x="phase", y="val", color="Situation", barmode="group", title="Value /100 balls by phase",
                                category_orders={"phase": ["Powerplay", "Middle", "Death"]}, color_discrete_sequence=[NAVY, PINK])),
                   use_container_width=True)
    bk = d.groupby("pi_bucket")["val"].mean().reindex(["Low", "Moderate", "Elevated", "High"]).reset_index()
    bk["val"] *= 100
    b.plotly_chart(style(px.bar(bk, x="pi_bucket", y="val", title="Value /100 balls by pressure level", color_discrete_sequence=[PINK])),
                   use_container_width=True)
    split_col = "pace_or_spin" if role == "bat" else "batting_type"
    if split_col in d and d["high_pressure"].any():
        sp = d[d["high_pressure"]].groupby(split_col).agg(balls=("val", "size"), value=("val", "mean")).reset_index()
        sp["value"] *= 100
        sp = sp[sp["balls"] >= 15]
        if len(sp):
            st.plotly_chart(style(px.bar(sp, x=split_col, y="value", text="balls",
                                         title="High-pressure value /100 balls vs " + ("pace / spin" if role == "bat" else "batter hand") + " (label = balls)",
                                         color_discrete_sequence=[PINK])), use_container_width=True)
    st.dataframe(row.drop(labels=[x for x in row.index if str(x).startswith("z_")]).astype(str).to_frame("value"), use_container_width=True)

# ------------------------------------------------------------------ 6 squad fit & auction
with T[5]:
    st.markdown("Players ranked by the model, with how each could strengthen the Royals. `RR_2026` marks players already in the IPL 2026 Rajasthan Royals squad.")
    role = st.radio("Role ", ["bat", "bowl"], horizontal=True, format_func=lambda r: "Batters" if r == "bat" else "Bowlers", key="fit_role")
    t = R[role]; e = t[t.eligible].copy()
    c1, c2, c3 = st.columns(3)
    roles = sorted(e["role"].dropna().unique())
    rsel = c1.multiselect("Role / phase", roles, default=roles)
    hide_rr = c2.checkbox("Hide current RR players", value=False)
    ipl_only = c3.checkbox("IPL 2025/26 squad players only", value=False)
    e = e[e["role"].isin(rsel)]
    if hide_rr: e = e[~e["RR_2026"].fillna(False)]
    if ipl_only: e = e[e["IPL2025"].fillna(False) | e["IPL2026"].fillna(False)]
    cols = ["rank", "player", "latest_team", "role", "RR_2026", "score", "fit"]
    st.dataframe(e[cols].head(30).assign(rank=lambda x: x["rank"].astype(int), score=lambda x: x["score"].round(2)),
                 use_container_width=True, hide_index=True, column_config={"fit": st.column_config.TextColumn("How he strengthens RR", width="large")})
    rr = t[t["RR_2026"].fillna(False)].sort_values("hp_balls", ascending=False)
    st.subheader("Rajasthan Royals squad: current pressure profile")
    if len(rr):
        rr_t = rr[["player", "hp_balls", "PV", "consistency", "eligible", "rank"]].copy()
        rr_t["rank"] = rr_t["rank"].where(rr_t["eligible"])
        st.dataframe(rr_t.round(2), use_container_width=True, hide_index=True)
    else:
        st.caption("No IPL 2026 Rajasthan Royals players found in the ratings for this filter.")
    if role == "bat":
        e2 = e.dropna(subset=["hp_SR_vs_pace", "hp_SR_vs_spin"]).head(15)
        if len(e2):
            fig = go.Figure()
            fig.add_bar(x=e2["player"], y=e2["hp_SR_vs_pace"], name="SR vs pace", marker_color=NAVY)
            fig.add_bar(x=e2["player"], y=e2["hp_SR_vs_spin"], name="SR vs spin", marker_color=PINK)
            fig.update_layout(barmode="group", title="High-pressure strike rate vs pace and spin (top candidates)")
            st.plotly_chart(style(fig), use_container_width=True)
    else:
        e2 = e.head(15)
        fig = go.Figure()
        for c_, nm, col in (("pp_share", "Powerplay", NAVY), ("mid_share", "Middle", GREY), ("death_share", "Death", PINK)):
            fig.add_bar(x=e2["player"], y=100 * e2[c_], name=nm, marker_color=col)
        fig.update_layout(barmode="stack", title="Where the top bowlers operate (% of deliveries)")
        st.plotly_chart(style(fig), use_container_width=True)

# ------------------------------------------------------------------ 7 reliability
with T[6]:
    st.markdown("**How much should we trust the rankings?** Pressure samples are small; these checks quantify it.")
    if st.button("Run reliability & sensitivity checks (≈30 s)"):
        for role, name in (("bat", "Batters"), ("bowl", "Bowlers")):
            rel = metrics.split_half_reliability(df, role, wicket_value=L)
            st.subheader(name)
            st.metric("Split-half Spearman ρ (random halves of matches)", f"{rel['mean']:.2f} ± {rel['sd']:.2f}")
            s, freq = sensitivity.top10_stability(df, role)
            st.dataframe(s, use_container_width=True, hide_index=True)
            nm_ = R[role].set_index("player_id")["player"]
            st.plotly_chart(style(px.bar(x=[nm_.get(i, i) for i in freq.index], y=freq.values * 100,
                                         labels={"x": "", "y": "% of scenarios still in top 10"},
                                         color_discrete_sequence=[PINK], title=f"{name}: robustness of each top-10 pick")),
                            use_container_width=True)

# ------------------------------------------------------------------ 8 submission
with T[7]:
    st.markdown("Pick these in the Unstop dropdowns (Rank 1 → 10). They match `python run_pipeline.py --report`.")
    a, b = st.columns(2)
    for role, col, label in (("bat", a, "Top-10 batters"), ("bowl", b, "Top-10 bowlers")):
        e = R[role][R[role].eligible].head(10)
        col.subheader(label)
        for i, p in enumerate(e["player"], 1):
            col.markdown(f"**{i}.** {p}")
        col.download_button(f"Download {label} (CSV)", e[["rank", "player", "PV", "resilience", "consistency", "score"]].round(3).to_csv(index=False),
                            f"top10_{role}.csv", key=f"sub_{role}")
    st.info("Unstop dropdown names can differ slightly (spelling/initials). Check each pick before submitting. "
            "The dashboard settings only change what you see here; run `python run_pipeline.py --report` to regenerate the report with the saved defaults in config.py.")
