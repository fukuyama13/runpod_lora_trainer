"""StatsBomb analysis tools.

All public functions return plain Python / JSON-serialisable objects
suitable for embedding in data.json and rendering on the dashboard.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

import statsbomb_loader as loader


# ── Helpers ─────────────────────────────────────────────────────────────────

def _safe(v, default=None):
    """Return *v* unless it is NaN / None."""
    try:
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    return v


def _fi(v, default=0) -> int:
    return int(_safe(v, default))


def _ff(v, default=0.0) -> float:
    return float(_safe(v, default))


# ── Catalogue ────────────────────────────────────────────────────────────────

def list_statsbomb_competitions() -> list[dict]:
    """Return all available competitions as a list of dicts."""
    df = loader.load_competitions()
    return [] if df.empty else df.to_dict("records")


def get_match_catalog(max_comps: int = 50) -> list[dict]:
    """
    Return a list of competitions, each containing their full match list.
    Capped at *max_comps* competition-seasons to keep the export size reasonable.
    """
    comps = loader.load_competitions()
    if comps.empty:
        return []

    catalog: list[dict] = []
    for _, c in comps.head(max_comps).iterrows():
        matches = loader.load_matches(int(c.competition_id), int(c.season_id))
        if matches.empty:
            continue
        match_list = [
            {
                "match_id":   int(m.match_id),
                "match_date": m.match_date,
                "home_team":  m.home_team,
                "away_team":  m.away_team,
                "home_score": int(m.home_score) if pd.notna(m.home_score) else None,
                "away_score": int(m.away_score) if pd.notna(m.away_score) else None,
                "stage":      m.stage,
            }
            for _, m in matches.iterrows()
        ]
        catalog.append({
            "competition_id": int(c.competition_id),
            "season_id":      int(c.season_id),
            "competition":    c.competition,
            "season":         c.season,
            "country":        c.country,
            "matches":        match_list,
        })
    return catalog


# ── Per-match tools ──────────────────────────────────────────────────────────

def get_shot_map(match_id: int) -> list[dict]:
    """All shots in a match with location, xG and outcome."""
    events = loader.load_events(match_id)
    if events.empty:
        return []
    return [
        {
            "player":    _safe(r.player,          "Unknown"),
            "team":      _safe(r.team,            "Unknown"),
            "minute":    _fi(r.minute),
            "x":         _ff(r.loc_x,  60.0),
            "y":         _ff(r.loc_y,  40.0),
            "xg":        _ff(getattr(r, "shot_xg", 0.05), 0.05),
            "outcome":   _safe(getattr(r, "shot_outcome",  None), "Unknown"),
            "technique": _safe(getattr(r, "shot_technique", None), ""),
            "body_part": _safe(getattr(r, "shot_body_part", None), ""),
        }
        for r in events[events["type"] == "Shot"].itertuples()
    ]


def get_pass_network(match_id: int, team_name: str) -> dict:
    """
    Pass network for *team_name*.
    Nodes = players at their average touch location.
    Edges = completed-pass combinations with count >= 2.
    """
    events = loader.load_events(match_id)
    if events.empty:
        return {"nodes": [], "edges": [], "team": team_name}

    te = events[events["team"] == team_name]

    pos = (
        te.dropna(subset=["loc_x", "loc_y", "player"])
        .groupby("player", as_index=False)
        .agg(avg_x=("loc_x", "mean"), avg_y=("loc_y", "mean"),
             touches=("index", "count"))
    )

    completed = te[
        (te["type"] == "Pass") &
        (te["pass_outcome"].isna()) &
        (te["pass_recipient"].notna())
    ]
    combos = (
        completed.groupby(["player", "pass_recipient"], as_index=False)
        .size().rename(columns={"size": "count"})
        .query("count >= 2")
    )

    return {
        "nodes": [{"player": r.player, "x": round(_ff(r.avg_x), 1),
                   "y": round(_ff(r.avg_y), 1), "touches": _fi(r.touches)}
                  for r in pos.itertuples()],
        "edges": [{"from": r.player, "to": r.pass_recipient, "count": _fi(r.count)}
                  for r in combos.itertuples()],
        "team":  team_name,
    }


def get_player_heatmap(match_id: int, player_name: str) -> dict:
    """All touch locations for *player_name* in *match_id*."""
    events = loader.load_events(match_id)
    if events.empty:
        return {"player": player_name, "touches": []}
    ev = events[
        (events["player"] == player_name) &
        events["loc_x"].notna() & events["loc_y"].notna()
    ]
    return {
        "player":  player_name,
        "touches": [{"x": _ff(r.loc_x), "y": _ff(r.loc_y),
                     "type": _safe(r.type, "Action"), "minute": _fi(r.minute)}
                    for r in ev.itertuples()],
    }


def get_match_stats(match_id: int) -> dict:
    """High-level match-stats breakdown for both teams."""
    events = loader.load_events(match_id)
    if events.empty:
        return {}
    stats: dict = {}
    for team in events["team"].dropna().unique():
        te  = events[events["team"] == team]
        sh  = te[te["type"] == "Shot"]
        pas = te[te["type"] == "Pass"]
        cp  = pas[pas["pass_outcome"].isna()]
        stats[str(team)] = {
            "shots":           len(sh),
            "shots_on_target": len(sh[sh["shot_outcome"].isin(["Saved", "Goal"])]),
            "goals":           len(sh[sh["shot_outcome"] == "Goal"]),
            "xg":              round(float(sh["shot_xg"].fillna(0).sum()), 2),
            "passes":          len(pas),
            "pass_accuracy":   round(len(cp) / max(len(pas), 1) * 100, 1),
            "dribbles":        len(te[te["type"] == "Dribble"]),
            "pressures":       len(te[te["type"] == "Pressure"]),
        }
    return stats


# ── Full active-match payload ────────────────────────────────────────────────

def build_active_match_payload(match_id: int) -> dict:
    """
    Build the complete event-data payload for one match.
    Returns a dict ready to be placed at data.statsbomb.active_match.
    """
    events = loader.load_events(match_id)
    if events.empty:
        return {"match_id": match_id, "error": "No events found for this match ID."}

    teams = events["team"].dropna().unique().tolist()
    home  = teams[0] if len(teams) > 0 else ""
    away  = teams[1] if len(teams) > 1 else ""

    # ── Shots ────────────────────────────────────────────────────────────────
    shots = [
        {
            "player":    _safe(r.player,          "Unknown"),
            "team":      _safe(r.team,            "Unknown"),
            "minute":    _fi(r.minute),
            "x":         _ff(r.loc_x,  60.0),
            "y":         _ff(r.loc_y,  40.0),
            "xg":        _ff(getattr(r, "shot_xg", 0.05), 0.05),
            "outcome":   _safe(getattr(r, "shot_outcome",  None), "Unknown"),
            "technique": _safe(getattr(r, "shot_technique", None), ""),
            "body_part": _safe(getattr(r, "shot_body_part", None), ""),
        }
        for r in events[events["type"] == "Shot"].itertuples()
    ]

    # ── Pass networks (both teams) ────────────────────────────────────────────
    pass_networks: dict = {}
    for team in teams:
        te = events[events["team"] == team]
        pos = (
            te.dropna(subset=["loc_x", "loc_y", "player"])
            .groupby("player", as_index=False)
            .agg(avg_x=("loc_x", "mean"), avg_y=("loc_y", "mean"),
                 touches=("index", "count"))
        )
        completed = te[
            (te["type"] == "Pass") &
            (te["pass_outcome"].isna()) &
            (te["pass_recipient"].notna())
        ]
        combos = (
            completed.groupby(["player", "pass_recipient"], as_index=False)
            .size().rename(columns={"size": "count"})
            .query("count >= 2")
        )
        pass_networks[str(team)] = {
            "nodes": [{"player": r.player, "x": round(_ff(r.avg_x), 1),
                       "y": round(_ff(r.avg_y), 1), "touches": _fi(r.touches)}
                      for r in pos.itertuples()],
            "edges": [{"from": r.player, "to": r.pass_recipient, "count": _fi(r.count)}
                      for r in combos.itertuples()],
        }

    # ── Player heatmaps (players with ≥ 10 touches) ───────────────────────────
    heatmaps: dict = {}
    for player in events["player"].dropna().unique():
        pe = events[
            (events["player"] == player) &
            events["loc_x"].notna() & events["loc_y"].notna()
        ]
        if len(pe) >= 10:
            heatmaps[str(player)] = [
                {"x": _ff(r.loc_x), "y": _ff(r.loc_y)}
                for r in pe.itertuples()
            ]

    # ── Match stats ───────────────────────────────────────────────────────────
    match_stats: dict = {}
    for team in teams:
        te  = events[events["team"] == team]
        sh  = te[te["type"] == "Shot"]
        pas = te[te["type"] == "Pass"]
        cp  = pas[pas["pass_outcome"].isna()]
        match_stats[str(team)] = {
            "shots":           len(sh),
            "shots_on_target": len(sh[sh["shot_outcome"].isin(["Saved", "Goal"])]),
            "goals":           len(sh[sh["shot_outcome"] == "Goal"]),
            "xg":              round(float(sh["shot_xg"].fillna(0).sum()), 2),
            "passes":          len(pas),
            "pass_accuracy":   round(len(cp) / max(len(pas), 1) * 100, 1),
            "dribbles":        len(te[te["type"] == "Dribble"]),
            "pressures":       len(te[te["type"] == "Pressure"]),
        }

    players_in_match = sorted(events["player"].dropna().unique().tolist())

    return {
        "match_id":     match_id,
        "home_team":    home,
        "away_team":    away,
        "shots":        shots,
        "pass_networks": pass_networks,
        "heatmaps":     heatmaps,
        "match_stats":  match_stats,
        "players":      players_in_match,
    }
