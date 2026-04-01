"""Load StatsBomb open-data from the cloned repository.

Clone the free data repo into ./statsbomb_data/ before using this module:
    git clone https://github.com/statsbomb/open-data.git statsbomb_data
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_SB_ROOT = Path(__file__).parent / "statsbomb_data" / "data"


# ── Availability ────────────────────────────────────────────────────────────

def is_available() -> bool:
    """Return True only when the open-data repo has been cloned."""
    return (_SB_ROOT / "competitions.json").exists()


# ── Low-level helpers ────────────────────────────────────────────────────────

def _jload(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ── Public loaders ───────────────────────────────────────────────────────────

def load_competitions() -> pd.DataFrame:
    """All competition / season combinations available in the repo."""
    if not is_available():
        return pd.DataFrame()
    raw = _jload(_SB_ROOT / "competitions.json")
    df = pd.json_normalize(raw).rename(columns={
        "competition_id":     "competition_id",
        "season_id":          "season_id",
        "competition_name":   "competition",
        "season_name":        "season",
        "country_name":       "country",
        "competition_gender": "gender",
    })
    keep = [c for c in ["competition_id", "season_id", "competition",
                         "season", "country", "gender"] if c in df.columns]
    return df[keep].copy()


def load_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """All matches for one competition + season, sorted by date."""
    path = _SB_ROOT / "matches" / str(competition_id) / f"{season_id}.json"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for m in _jload(path):
        rows.append({
            "match_id":   m["match_id"],
            "match_date": m.get("match_date", ""),
            "kick_off":   m.get("kick_off", ""),
            "home_team":  m["home_team"]["home_team_name"],
            "away_team":  m["away_team"]["away_team_name"],
            "home_score": m.get("home_score"),
            "away_score": m.get("away_score"),
            "stage":      (m.get("competition_stage") or {}).get("name", ""),
            "stadium":    (m.get("stadium") or {}).get("name", ""),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("match_date").reset_index(drop=True) if not df.empty else df


def load_events(match_id: int) -> pd.DataFrame:
    """All events for a match, flattened into one row per event."""
    path = _SB_ROOT / "events" / f"{match_id}.json"
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([_flatten_event(e) for e in _jload(path)])


def load_lineups(match_id: int) -> pd.DataFrame:
    """Starting lineups (both teams) for a match."""
    path = _SB_ROOT / "lineups" / f"{match_id}.json"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for td in _jload(path):
        for p in td.get("lineup", []):
            pos_list = p.get("positions", [])
            rows.append({
                "team":          td["team_name"],
                "player_id":     p["player_id"],
                "player_name":   p["player_name"],
                "jersey_number": p.get("jersey_number"),
                "position":      pos_list[0]["position"] if pos_list else "Unknown",
                "country":       (p.get("country") or {}).get("name", ""),
            })
    return pd.DataFrame(rows)


# ── Event flattener ──────────────────────────────────────────────────────────

def _flatten_event(e: dict) -> dict:
    loc = e.get("location") or []
    row: dict = {
        "id":     e.get("id"),
        "index":  e.get("index"),
        "period": e.get("period"),
        "minute": e.get("minute"),
        "second": e.get("second"),
        "type":   (e.get("type")   or {}).get("name"),
        "team":   (e.get("team")   or {}).get("name"),
        "player": (e.get("player") or {}).get("name"),
        "loc_x":  loc[0] if len(loc) > 0 else None,
        "loc_y":  loc[1] if len(loc) > 1 else None,
    }

    if "shot" in e:
        s  = e["shot"]
        el = s.get("end_location") or []
        row.update({
            "shot_xg":        s.get("statsbomb_xg"),
            "shot_outcome":   (s.get("outcome")   or {}).get("name"),
            "shot_technique": (s.get("technique") or {}).get("name"),
            "shot_body_part": (s.get("body_part")  or {}).get("name"),
            "shot_end_x":     el[0] if len(el) > 0 else None,
            "shot_end_y":     el[1] if len(el) > 1 else None,
        })

    if "pass" in e:
        p  = e["pass"]
        el = p.get("end_location") or []
        row.update({
            "pass_recipient": (p.get("recipient") or {}).get("name"),
            "pass_outcome":   (p.get("outcome")   or {}).get("name"),  # None = complete
            "pass_length":    p.get("length"),
            "pass_angle":     p.get("angle"),
            "pass_cross":     p.get("cross", False),
            "pass_switch":    p.get("switch", False),
            "pass_end_x":     el[0] if len(el) > 0 else None,
            "pass_end_y":     el[1] if len(el) > 1 else None,
        })

    if "carry" in e:
        el = e["carry"].get("end_location") or []
        row["carry_end_x"] = el[0] if len(el) > 0 else None
        row["carry_end_y"] = el[1] if len(el) > 1 else None

    if "dribble" in e:
        row["dribble_outcome"] = (e["dribble"].get("outcome") or {}).get("name")

    return row
