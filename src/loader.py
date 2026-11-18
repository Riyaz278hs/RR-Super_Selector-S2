"""Loading + schema normalisation.

The ball-by-ball file can come with many column spellings (Cricsheet-like, CricAPI-like, custom).
`detect_schema` guesses the mapping from aliases; the dashboard lets the user override any guess.
`normalise_bbb` converts to one canonical frame used by everything else.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
import config

# canonical field -> list of aliases (lower-case, non-alphanumerics stripped)
ALIASES = {
    "match_id": ["matchid", "match", "gameid", "id"],
    "innings": ["innings", "inning", "inns", "inningsno", "inningsnumber", "inn"],
    "over": ["over", "overs", "overno", "overnumber", "overnum", "overid"],
    "seq": ["inningsballnum", "inningsballno", "ballseq", "deliveryseq", "inningsball"],
    "extras2": ["extras2runs", "extra2runs"],
    "ball": ["ball", "ballno", "ballnumber", "ballinover", "delivery"],
    "batter": ["batterid", "batsmanid", "strikerid", "batter", "batsman", "striker", "batsmanname", "battername"],
    "bowler": ["bowlerid", "bowler", "bowlername"],
    "player_out": ["playeroutid", "playerdismissedid", "dismissedplayerid", "playerout", "playerdismissed", "dismissedplayer", "outbatsmanid", "dismissed"],
    "runs_bat": ["runsbatter", "batterruns", "batsmanruns", "runsoffbat", "batruns", "runsbat", "runs", "runsscored"],
    "extras": ["extras", "extraruns", "extra", "totalextras"],
    "total_runs": ["totalruns", "runstotal", "total", "ballruns", "runsonball"],
    "wides": ["wides", "wide", "widesruns", "iswide"],
    "noballs": ["noballs", "noball", "noballsruns", "isnoball"],
    "byes": ["byes", "bye", "isbye"],
    "legbyes": ["legbyes", "legbye", "islegbye"],
    "extra_type": ["extratype", "extrastype", "extrakind"],
    "is_wicket": ["iswicket", "wicket", "wicketfell", "isout", "wickets", "dismissal"],
    "wicket_type": ["wickettype", "dismissaltype", "dismissalkind", "wicketkind", "howout", "outtype"],
}

NON_BOWLER_PATTERN = r"run ?out|retire|obstruct|obstract|timed|handled"


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def find_bbb_file(data_dir: Path | None = None) -> Path | None:
    data_dir = Path(data_dir or config.DATA_DIR)
    for n in config.BBB_CANDIDATES:
        if (data_dir / n).exists():
            return data_dir / n
    known = {config.F_COMPETITIONS, config.F_MATCHES, config.F_PLAYING11, config.F_MAPPING}
    others = [p for p in data_dir.glob("*.csv") if p.name not in known and not p.name.startswith("sample_")]
    others.sort(key=lambda p: p.stat().st_size, reverse=True)
    if others:
        return others[0]
    samples = list(data_dir.glob("sample_*.csv"))
    return samples[0] if samples else None


def detect_schema(columns) -> dict:
    """Return {canonical: actual_column or None}."""
    keys = {_key(c): c for c in columns}
    out = {}
    for canon, aliases in ALIASES.items():
        out[canon] = next((keys[a] for a in aliases if a in keys), None)
    # prefer *_id style columns for batter/bowler/player_out when both id and name exist
    for canon in ("batter", "bowler", "player_out"):
        ids = [c for c in columns if _key(c).endswith("id") and canon.split("_")[0] in _key(c)]
        if ids:
            out[canon] = ids[0]
    return out


def load_supporting(data_dir: Path | None = None) -> dict:
    d = Path(data_dir or config.DATA_DIR)
    comps = pd.read_csv(d / config.F_COMPETITIONS)
    matches = pd.read_csv(d / config.F_MATCHES, parse_dates=["match_date"])
    xi = pd.read_csv(d / config.F_PLAYING11)
    mapping = pd.read_csv(d / config.F_MAPPING)
    return dict(competitions=comps, matches=matches, playing11=xi, mapping=mapping)


def player_lookup(mapping: pd.DataFrame) -> pd.DataFrame:
    """player_id -> name (+ latest team & whether seen in 2026)."""
    names = mapping.drop_duplicates("player_id")[["player_id", "player_name"]]
    return names.set_index("player_id")["player_name"]


def _to_num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0)


def _to_bool(s):
    if s.dtype == bool:
        return s
    t = s.astype(str).str.strip().str.lower()
    return t.isin(["1", "true", "t", "yes", "y", "1.0"])


def normalise_bbb(raw: pd.DataFrame, schema: dict, sup: dict) -> tuple[pd.DataFrame, dict]:
    """Canonical ball frame. Returns (df, info) where info logs assumptions made."""
    info = {"notes": []}
    missing = [k for k in ("match_id", "innings", "batter", "bowler") if not schema.get(k)]
    if missing:
        raise ValueError(f"Could not map required columns: {missing}. Columns found: {list(raw.columns)}")

    df = pd.DataFrame({"match_id": pd.to_numeric(raw[schema["match_id"]], errors="coerce")})
    inn = raw[schema["innings"]]
    if inn.dtype == object:
        inn = inn.astype(str).str.extract(r"(\d+)")[0]
    df["innings"] = pd.to_numeric(inn, errors="coerce")
    df["file_order"] = pd.to_numeric(raw[schema["seq"]], errors="coerce").values if schema.get("seq") else np.arange(len(raw))
    for c in ("batting_order", "batting_type", "bowling_type", "pace_or_spin", "batting_team_id", "bowling_team_id"):
        if c in raw.columns:                      # optional player-profile columns (used for squad-fit insights)
            df[c] = raw[c].values
    if schema.get("over"):
        df["over_raw"] = pd.to_numeric(raw[schema["over"]], errors="coerce")
    for k in ("batter", "bowler"):
        df[k] = raw[schema[k]]
    df["player_out"] = raw[schema["player_out"]] if schema.get("player_out") else np.nan

    # --- runs & extras ---
    bat = _to_num(raw[schema["runs_bat"]]) if schema.get("runs_bat") else None
    tot = _to_num(raw[schema["total_runs"]]) if schema.get("total_runs") else None
    ext = _to_num(raw[schema["extras"]]) if schema.get("extras") else None
    if bat is None and tot is not None and ext is not None:
        bat = tot - ext
    if tot is None and bat is not None and ext is not None:
        tot = bat + ext
    if ext is None and bat is not None and tot is not None:
        ext = tot - bat
    if bat is not None and tot is None and ext is None and schema.get("extra_type"):
        ext = pd.Series(0, index=raw.index); tot = bat          # amounts estimated below from extra_type
    if bat is None or tot is None:
        raise ValueError("Need runs off the bat and (extras or total runs).")
    df["runs_bat"], df["runs_total"], df["extras"] = bat.values, tot.values, (tot - bat).values

    def extra_col(canon):
        c = schema.get(canon)
        return _to_num(raw[c]).values if c else np.zeros(len(raw))
    wide, nb, bye, lb = (extra_col(x) for x in ("wides", "noballs", "byes", "legbyes"))
    est_note = False
    if schema.get("extra_type"):
        t = raw[schema["extra_type"]].astype(str).str.lower().str.strip()
        is_w = t.str.contains("wide"); is_nb = t.str.contains("no ?ball|overstep|other"); is_b = t.str.fullmatch(r"byes?")
        is_lb = t.str.contains("leg"); is_pen = t.str.contains("penalt")
        if (df["extras"] == 0).all():
            # no amount columns: estimate (wide/no-ball/bye/leg-bye = 1 run, penalty = 5) + any second-extras column
            e2 = _to_num(raw[schema["extras2"]]).values if schema.get("extras2") else 0
            ext_est = np.select([is_w, is_nb, is_b, is_lb, is_pen], [1, 1, 1, 1, 5], 0) + e2
            ext = pd.Series(ext_est, index=raw.index); tot = bat + ext
            df["runs_bat"], df["runs_total"], df["extras"] = bat.values, tot.values, ext.values
            est_note = True
            info["notes"].append("Extras amounts not in file: wides/no-balls/byes/leg-byes estimated at 1 run each (validated against match margins: median error 1-2 runs per match).")
        wide = np.where(is_w, df["extras"], wide)
        nb = np.where(is_nb, np.maximum(df["extras"], 1), nb)
        bye = np.where(is_b, df["extras"], bye)
        lb = np.where(is_lb, df["extras"], lb)
    df["is_wide"] = wide > 0
    df["is_noball"] = nb > 0
    df["legal"] = ~(df["is_wide"] | df["is_noball"])
    df["faced"] = ~df["is_wide"]                       # batter faces no-balls, not wides
    # bowler is charged bat runs + wides + no-balls, but not byes / leg-byes
    df["runs_bowler"] = (df["runs_bat"] + df["extras"] - bye - lb).clip(lower=0)

    # --- wickets ---
    wt = raw[schema["wicket_type"]].astype(str).str.lower().str.strip() if schema.get("wicket_type") else pd.Series("", index=raw.index)
    wt = wt.replace({"nan": "", "none": ""})
    if schema.get("is_wicket"):
        w = raw[schema["is_wicket"]]
        wk = _to_bool(w) if w.dtype in (object, bool) and not np.issubdtype(w.dtype, np.number) else (pd.to_numeric(w, errors="coerce").fillna(0) > 0)
    elif schema.get("wicket_type"):
        wk = wt != ""
    elif schema.get("player_out"):
        wk = raw[schema["player_out"]].notna() & (raw[schema["player_out"]].astype(str).str.strip() != "")
    else:
        raise ValueError("No wicket information (is_wicket / wicket_type / player_out) found.")
    df["is_wicket"] = wk.values
    df["wicket_type"] = wt.values
    df["bowler_wicket"] = df["is_wicket"] & ~wt.str.contains(NON_BOWLER_PATTERN).values
    df["is_retired"] = wt.str.contains("retired").values
    # batter dismissed on this ball?
    if schema.get("player_out"):
        po_n = pd.to_numeric(df["player_out"], errors="coerce")
        bt_n = pd.to_numeric(df["batter"], errors="coerce")
        if po_n.notna().sum() > 0 and bt_n.notna().mean() > 0.9:      # id columns
            same = po_n == bt_n
        else:                                                       # name columns
            same = df["player_out"].astype(str).str.strip() == df["batter"].astype(str).str.strip()
        df["batter_out"] = df["is_wicket"] & ~df["is_retired"] & same
        n_w = max(int(df["is_wicket"].sum()), 1)
        if (df["is_wicket"] & ~df["is_retired"] & ~same).sum() / n_w > 0.3:
            info["notes"].append("More than 30% of wickets dismiss someone other than the striker; check the player_out mapping.")
    else:
        df["batter_out"] = df["is_wicket"] & ~df["is_retired"]
        info["notes"].append("No player_out column: every wicket assumed to dismiss the striker.")

    df = df.dropna(subset=["match_id", "innings"])
    df["match_id"] = df["match_id"].astype(int)
    df["innings"] = df["innings"].astype(int)
    n0 = len(df)
    df = df[df["innings"].isin([1, 2])]
    if len(df) < n0:
        info["notes"].append(f"Dropped {n0 - len(df):,} rows from innings >2 (super overs).")

    # --- attach match context ---
    m = sup["matches"].merge(sup["competitions"][["comp_id", "comp_name", "match_format", "season"]], on="comp_id", how="left")
    m["revised"] = m[["first_inn_revised_overs", "second_inn_revised_overs", "revised_target"]].notna().any(axis=1)
    keep = ["match_id", "comp_id", "comp_name", "match_format", "season", "match_date", "revised", "win_team_id", "stage", "knock_out"]
    df = df.merge(m[keep], on="match_id", how="left")
    miss = df["match_format"].isna().mean()
    if miss > 0:
        info["notes"].append(f"{miss:.1%} of balls belong to matches not present in matches.csv (dropped).")
        df = df[df["match_format"].notna()]
    if config.EXCLUDE_REVISED:
        nr = df["revised"].sum()
        if nr:
            info["notes"].append(f"Excluded {df.loc[df['revised'], 'match_id'].nunique()} rain-revised matches ({nr:,} balls).")
        df = df[~df["revised"]]
    df["tournament"] = np.where(df["comp_name"].str.contains("IPL", na=False), "IPL", "T20I")

    # --- ids <-> names ---
    names = player_lookup(sup["mapping"])
    for k in ("batter", "bowler"):
        if pd.api.types.is_numeric_dtype(df[k]):
            df[k] = df[k].astype("Int64")
            df[k + "_name"] = df[k].map(names)
        else:
            name2id = {v: k2 for k2, v in names.items()}
            df[k + "_name"] = df[k]
            df[k] = df[k].map(name2id)
            if df[k].isna().mean() > 0.05:
                info["notes"].append(f"{k}: {df[k].isna().mean():.0%} of names could not be matched to player_id; names used as keys.")
                df[k] = df[k + "_name"]
    df[["batter_name", "bowler_name"]] = df[["batter_name", "bowler_name"]].fillna("Unknown")
    if pd.api.types.is_numeric_dtype(df["player_out"]):
        df["player_out"] = df["player_out"].astype("Int64")

    # --- ball counters (file order within match & innings) ---
    df = df.sort_values(["match_id", "innings", "file_order"], kind="stable").reset_index(drop=True)
    g = df.groupby(["match_id", "innings"], sort=False)
    df["legal_cum"] = g["legal"].cumsum()
    df["balls_before"] = (df["legal_cum"] - df["legal"].astype(int)).clip(upper=119)   # legal balls bowled BEFORE this delivery
    df["wkts_before"] = g["is_wicket"].cumsum() - df["is_wicket"].astype(int)
    df["score_before"] = g["runs_total"].cumsum() - df["runs_total"]
    df["over_idx"] = (df["balls_before"] // 6).astype(int)
    df["phase"] = pd.cut(df["over_idx"], [-1, 5, 14, 19], labels=["Powerplay", "Middle", "Death"]).astype(str)
    return df, info
