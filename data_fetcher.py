"""
Fetches real Premier League data from the football-data.org v4 API.

All responses are cached as JSON under api_cache/ so the API is only
hit once per session.  Delete the cache folder to force a refresh.

Free-tier limits: 10 req/min, no xG/shot data (those are simulated).
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# Load .env once at import; individual functions re-read so a key added
# after import is still picked up on the next call.
load_dotenv()

BASE_URL    = "https://api.football-data.org/v4"
COMPETITION = "PL"

CACHE_DIR = Path("api_cache")
CACHE_DIR.mkdir(exist_ok=True)


def _api_key() -> str:
    """Return the API key, always checking .env directly so a freshly-created
    or updated .env is picked up without restarting the process."""
    # 1. Read .env file directly (bypasses stale os.environ cache)
    env_file = Path(".env")
    if env_file.exists():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == "FOOTBALL_API_KEY":
                val = v.strip()
                if val and val != "your_key_here":
                    return val
    # 2. Fall back to whatever is already in the environment
    return os.getenv("FOOTBALL_API_KEY", "")

# ---------------------------------------------------------------------------
# Team name normalisation
# ---------------------------------------------------------------------------
_TEAM_NAME_MAP: dict[str, str] = {
    "Manchester City FC":          "Man City",
    "Manchester United FC":        "Man United",
    "Nottingham Forest FC":        "Nottm Forest",
    "Tottenham Hotspur FC":        "Tottenham",
    "Newcastle United FC":         "Newcastle",
    "Brighton & Hove Albion FC":   "Brighton",
    "West Ham United FC":          "West Ham",
    "Wolverhampton Wanderers FC":  "Wolves",
    "Sheffield United FC":         "Sheffield Utd",
    "Luton Town FC":               "Luton",
    "AFC Bournemouth":             "Bournemouth",
    "Brentford FC":                "Brentford",
    "Fulham FC":                   "Fulham",
    "Everton FC":                  "Everton",
    "Crystal Palace FC":           "Crystal Palace",
    "Aston Villa FC":              "Aston Villa",
    "Arsenal FC":                  "Arsenal",
    "Chelsea FC":                  "Chelsea",
    "Liverpool FC":                "Liverpool",
    "Burnley FC":                  "Burnley",
    "Leicester City FC":           "Leicester",
    "Ipswich Town FC":             "Ipswich",
    "Southampton FC":              "Southampton",
}

_POSITION_MAP: dict[str, str] = {
    "Goalkeeper":          "GK",
    "Centre-Back":         "CB",
    "Left-Back":           "LB",
    "Right-Back":          "RB",
    "Defensive Midfield":  "CDM",
    "Central Midfield":    "CM",
    "Attacking Midfield":  "CAM",
    "Left Winger":         "LW",
    "Right Winger":        "RW",
    "Centre-Forward":      "ST",
    "Left Midfield":       "CM",
    "Right Midfield":      "CM",
    "Offence":             "ST",
    "Midfield":            "CM",
    "Defence":             "CB",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_configured() -> bool:
    """Return True when a real API key has been provided in .env."""
    key = _api_key()
    return bool(key and key.strip() not in ("", "your_key_here"))


def normalise_team(full_name: str, short_name: str = "") -> str:
    return _TEAM_NAME_MAP.get(full_name, short_name or full_name)


def normalise_position(pos: str) -> str:
    return _POSITION_MAP.get(pos, "CM")


# ---------------------------------------------------------------------------
# Internal HTTP + cache layer
# ---------------------------------------------------------------------------

def _get(
    endpoint: str,
    params: dict[str, Any] | None = None,
    cache_key: str | None = None,
) -> dict[str, Any]:
    """HTTP GET with optional local JSON cache."""
    if cache_key:
        cache_file = CACHE_DIR / f"{cache_key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

    url = f"{BASE_URL}{endpoint}"
    resp = requests.get(
        url,
        headers={"X-Auth-Token": _api_key()},
        params=params or {},
        timeout=20,
    )
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()

    if cache_key:
        cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return data


# ---------------------------------------------------------------------------
# Public fetch functions
# ---------------------------------------------------------------------------

def fetch_standings() -> list[dict[str, Any]]:
    """Return one dict per team with current PL season standings."""
    raw = _get(
        f"/competitions/{COMPETITION}/standings",
        cache_key="pl_standings",
    )
    total_table = next(
        t["table"] for t in raw["standings"] if t["type"] == "TOTAL"
    )
    result = []
    for row in total_table:
        result.append({
            "team":         normalise_team(
                                row["team"]["name"],
                                row["team"].get("shortName", ""),
                            ),
            "team_id":      row["team"]["id"],
            "position":     row["position"],
            "played":       row["playedGames"],
            "won":          row["won"],
            "drawn":        row["draw"],
            "lost":         row["lost"],
            "points":       row["points"],
            "goals_for":    row["goalsFor"],
            "goals_against": row["goalsAgainst"],
            "goal_diff":    row["goalDifference"],
        })
    return result


def fetch_scorers(limit: int = 20) -> list[dict[str, Any]]:
    """Return top scorer dicts for the current PL season."""
    raw = _get(
        f"/competitions/{COMPETITION}/scorers",
        params={"limit": limit},
        cache_key=f"pl_scorers_{limit}",
    )
    result = []
    for s in raw.get("scorers", []):
        player = s["player"]
        team   = s.get("team", {})

        dob = player.get("dateOfBirth", "2000-01-01")
        try:
            age = date.today().year - int(dob[:4])
        except Exception:
            age = 25

        result.append({
            "player_id":      player["id"],
            "name":           player["name"],
            "team":           normalise_team(
                                  team.get("name", ""),
                                  team.get("shortName", ""),
                              ),
            "team_id":        team.get("id"),
            "position":       normalise_position(player.get("position", "")),
            "nationality":    player.get("nationality", "Unknown"),
            "age":            age,
            "goals":          s.get("goals") or 0,
            "assists":        s.get("assists") or 0,
            "penalties":      s.get("penalties") or 0,
            "played_matches": s.get("playedMatches") or 0,
        })
    return result


def fetch_matches() -> list[dict[str, Any]]:
    """Return all finished PL match results for the current season."""
    raw = _get(
        f"/competitions/{COMPETITION}/matches",
        params={"status": "FINISHED"},
        cache_key="pl_matches",
    )
    result = []
    for m in raw.get("matches", []):
        score = m.get("score", {}).get("fullTime", {})
        hg = score.get("home")
        ag = score.get("away")
        if hg is None or ag is None:
            continue

        result.append({
            "match_id":  m["id"],
            "matchweek": m.get("matchday") or 0,
            "home_team": normalise_team(
                             m["homeTeam"].get("name", ""),
                             m["homeTeam"].get("shortName", ""),
                         ),
            "away_team": normalise_team(
                             m["awayTeam"].get("name", ""),
                             m["awayTeam"].get("shortName", ""),
                         ),
            "home_goals": int(hg),
            "away_goals": int(ag),
        })
    return result


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not is_configured():
        print("No API key found.")
        print("Edit .env and set FOOTBALL_API_KEY=<your_key_from_football-data.org>")
    else:
        s = fetch_standings()
        m = fetch_matches()
        sc = fetch_scorers(10)
        print(f"Standings : {len(s)} teams")
        print(f"Matches   : {len(m)} finished")
        print(f"Scorers   : {len(sc)} players")
        print(f"Leader    : {s[0]['team']} ({s[0]['points']} pts)")
        print(f"Top scorer: {sc[0]['name']} ({sc[0]['goals']} goals)")
