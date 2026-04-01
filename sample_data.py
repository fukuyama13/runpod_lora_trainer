"""
Data layer for football analytics.

Priority order:
  1. Real data  – fetched from football-data.org when FOOTBALL_API_KEY is set.
  2. Synthetic  – fully simulated season used as a fallback.

The three public functions (generate_players, generate_matches,
generate_player_match_stats) always return DataFrames with identical
column schemas so the analytics layer never needs to know which path ran.

Real data provides:
  • Actual team names and league standings
  • Exact match scores (home_goals / away_goals)
  • Top-scorer names, goals, assists
Missing fields (xG, shots, possession, player attributes) are estimated /
simulated even in real-data mode — they are not available on the free tier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level state (populated by generate_players)
# ---------------------------------------------------------------------------

TEAMS: list[str] = []          # set at runtime; used by analytics layer

# ---------------------------------------------------------------------------
# Constants for synthetic generation
# ---------------------------------------------------------------------------

_FALLBACK_TEAMS = [
    "Arsenal", "Chelsea", "Liverpool", "Man City", "Man United",
    "Tottenham", "Newcastle", "Aston Villa", "Brighton", "West Ham",
    "Brentford", "Fulham", "Crystal Palace", "Wolves", "Everton",
    "Nottm Forest", "Bournemouth", "Burnley", "Sheffield Utd", "Luton",
]

POSITIONS = ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST"]
_ATTACKER_POSITIONS = {"ST", "LW", "RW", "CAM"}

_FIRST_NAMES = [
    "James", "Marcus", "Harry", "Bukayo", "Phil", "Erling", "Mohamed",
    "Bruno", "Kevin", "Virgil", "Trent", "Declan", "Mason", "Jack",
    "Ollie", "Raheem", "Jadon", "Ben", "Jordan", "Luke",
]
_LAST_NAMES = [
    "Smith", "Jones", "Williams", "Taylor", "Brown", "Davies", "Evans",
    "Wilson", "Thomas", "Roberts", "Walker", "White", "Hall", "Green",
    "Lewis", "Martin", "Clarke", "Anderson", "Wright", "Thompson",
]
_NATIONALITIES = [
    "England", "France", "Spain", "Brazil", "Argentina",
    "Portugal", "Germany", "Netherlands", "Belgium", "Norway",
]

_PLAYERS_PER_TEAM = 22
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Real-data loader (lazy, cached in module)
# ---------------------------------------------------------------------------

_real_cache: dict | None = None
_real_attempted = False


def _load_real_data() -> dict | None:
    """Try once to fetch real data; returns None on any failure."""
    global _real_cache, _real_attempted
    if _real_attempted:
        return _real_cache
    _real_attempted = True

    try:
        import data_fetcher
        if not data_fetcher.is_configured():
            return None
        print("  Connecting to football-data.org …")
        standings = data_fetcher.fetch_standings()
        matches   = data_fetcher.fetch_matches()
        scorers   = data_fetcher.fetch_scorers(20)
        if not standings or not matches:
            return None
        _real_cache = {
            "standings": standings,
            "matches":   matches,
            "scorers":   scorers,
        }
        print(f"  Real data: {len(standings)} teams | "
              f"{len(matches)} matches | {len(scorers)} scorers")
        return _real_cache
    except Exception as exc:
        print(f"  [Warning] Real data unavailable ({exc}). Using synthetic data.")
        return None


# ---------------------------------------------------------------------------
# Attribute generation helpers
# ---------------------------------------------------------------------------

def _position_attrs(position: str) -> dict[str, int]:
    if position == "GK":
        return dict(
            finishing   = int(RNG.integers(20, 45)),
            passing     = int(RNG.integers(50, 75)),
            defending   = int(RNG.integers(55, 80)),
            physicality = int(RNG.integers(60, 85)),
            pace        = int(RNG.integers(40, 65)),
        )
    if position in ("CB", "LB", "RB"):
        return dict(
            finishing   = int(RNG.integers(25, 55)),
            passing     = int(RNG.integers(55, 80)),
            defending   = int(RNG.integers(65, 90)),
            physicality = int(RNG.integers(65, 88)),
            pace        = int(RNG.integers(60, 85)),
        )
    if position in ("CDM", "CM"):
        return dict(
            finishing   = int(RNG.integers(40, 70)),
            passing     = int(RNG.integers(65, 90)),
            defending   = int(RNG.integers(50, 78)),
            physicality = int(RNG.integers(60, 85)),
            pace        = int(RNG.integers(58, 80)),
        )
    if position == "CAM":
        return dict(
            finishing   = int(RNG.integers(55, 82)),
            passing     = int(RNG.integers(70, 92)),
            defending   = int(RNG.integers(30, 55)),
            physicality = int(RNG.integers(55, 78)),
            pace        = int(RNG.integers(65, 88)),
        )
    # LW / RW / ST
    return dict(
        finishing   = int(RNG.integers(65, 95)),
        passing     = int(RNG.integers(60, 85)),
        defending   = int(RNG.integers(25, 50)),
        physicality = int(RNG.integers(58, 82)),
        pace        = int(RNG.integers(70, 96)),
    )


def _overall(attrs: dict[str, int]) -> int:
    return int(
        attrs["finishing"]   * 0.25
        + attrs["passing"]   * 0.20
        + attrs["defending"] * 0.20
        + attrs["physicality"] * 0.15
        + attrs["pace"]      * 0.20
    )


def _random_name() -> str:
    return f"{RNG.choice(_FIRST_NAMES)} {RNG.choice(_LAST_NAMES)}"


# ---------------------------------------------------------------------------
# PUBLIC: generate_players
# ---------------------------------------------------------------------------

def generate_players(players_per_team: int = _PLAYERS_PER_TEAM) -> pd.DataFrame:
    """
    Return a players DataFrame.

    Real mode: uses actual PL team names; injects real top scorers as the
               primary attacker slots per team (with simulated attributes).
    Synthetic: full random generation with fallback team names.
    """
    global TEAMS
    real = _load_real_data()

    if real:
        teams   = [s["team"] for s in real["standings"]]
        scorers = real["scorers"]
    else:
        teams   = _FALLBACK_TEAMS
        scorers = []

    TEAMS = teams

    # Index scorers by team for quick lookup
    scorers_by_team: dict[str, list[dict]] = {}
    for sc in scorers:
        scorers_by_team.setdefault(sc["team"], []).append(sc)

    rows: list[dict] = []
    player_id = 1
    # Track IDs assigned to real scorers so player_stats can reference them
    _scorer_id_map: dict[str, int] = {}   # scorer name → player_id

    for team in teams:
        team_scorers = scorers_by_team.get(team, [])
        attacker_slot = 0   # index into team_scorers

        for i in range(players_per_team):
            position = POSITIONS[i % len(POSITIONS)]
            attrs    = _position_attrs(position)
            overall  = _overall(attrs)

            # Inject a real scorer into this attacker slot
            real_sc = None
            if position in _ATTACKER_POSITIONS and attacker_slot < len(team_scorers):
                real_sc = team_scorers[attacker_slot]
                attacker_slot += 1

            if real_sc:
                name        = real_sc["name"]
                nationality = real_sc.get("nationality") or RNG.choice(_NATIONALITIES)
                age         = real_sc.get("age") or int(RNG.integers(22, 32))
                _scorer_id_map[name] = player_id
            else:
                name        = _random_name()
                nationality = RNG.choice(_NATIONALITIES)
                age         = int(RNG.integers(18, 35))

            rows.append({
                "player_id":   player_id,
                "name":        name,
                "team":        team,
                "position":    position,
                "nationality": nationality,
                "age":         age,
                "overall":     overall,
                **attrs,
            })
            player_id += 1

    df = pd.DataFrame(rows)
    # Store the scorer→id mapping on the module for use in player-stats generation
    global _SCORER_ID_MAP
    _SCORER_ID_MAP = _scorer_id_map
    return df


_SCORER_ID_MAP: dict[str, int] = {}


# ---------------------------------------------------------------------------
# PUBLIC: generate_matches
# ---------------------------------------------------------------------------

def generate_matches(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a matches DataFrame.

    Real mode: uses actual match results for goals; estimates xG / shots /
               possession / cards from those goals using a statistical model.
    Synthetic: full Poisson-based simulation driven by team strength.
    """
    real = _load_real_data()

    if real:
        return _matches_from_real(real["matches"])
    return _matches_synthetic(players_df)


def _xg_from_goals(goals: int) -> float:
    """Estimate xG from a known scoreline using a lognormal perturbation."""
    base = max(0.3, float(goals))
    return round(float(np.clip(base + RNG.normal(0, 0.55), 0.15, 6.0)), 2)


def _matches_from_real(real_matches: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for m in real_matches:
        hg = m["home_goals"]
        ag = m["away_goals"]

        home_xg   = _xg_from_goals(hg)
        away_xg   = _xg_from_goals(ag)
        home_shots = int(max(2, home_xg * 5.5 + RNG.integers(1, 7)))
        away_shots = int(max(2, away_xg * 5.5 + RNG.integers(1, 7)))
        poss = round(float(np.clip(50 + RNG.normal(0, 7), 33, 67)), 1)

        rows.append({
            "match_id":              m["match_id"],
            "matchweek":             m["matchweek"],
            "home_team":             m["home_team"],
            "away_team":             m["away_team"],
            "home_goals":            hg,
            "away_goals":            ag,
            "home_xg":               home_xg,
            "away_xg":               away_xg,
            "home_shots":            home_shots,
            "away_shots":            away_shots,
            "home_shots_on_target":  max(1, int(home_shots * RNG.uniform(0.30, 0.55))),
            "away_shots_on_target":  max(1, int(away_shots * RNG.uniform(0.25, 0.50))),
            "home_possession":       poss,
            "away_possession":       round(100 - poss, 1),
            "home_passes":           int(RNG.integers(300, 650)),
            "away_passes":           int(RNG.integers(270, 600)),
            "home_fouls":            int(RNG.integers(6, 18)),
            "away_fouls":            int(RNG.integers(6, 18)),
            "home_yellow_cards":     int(RNG.integers(0, 4)),
            "away_yellow_cards":     int(RNG.integers(0, 4)),
            "home_red_cards":        int(RNG.binomial(1, 0.05)),
            "away_red_cards":        int(RNG.binomial(1, 0.05)),
        })
    return pd.DataFrame(rows)


def _simulate_match(home: str, away: str, players_df: pd.DataFrame) -> dict:
    home_str = players_df[players_df["team"] == home]["overall"].mean()
    away_str = players_df[players_df["team"] == away]["overall"].mean()

    home_xg = max(0.3, (home_str / away_str) * 1.4 + RNG.normal(0, 0.4))
    away_xg = max(0.3, (away_str / home_str) * 1.0 + RNG.normal(0, 0.4))

    hg = int(RNG.poisson(home_xg))
    ag = int(RNG.poisson(away_xg))

    home_shots = int(home_xg * 6 + RNG.integers(2, 8))
    away_shots = int(away_xg * 6 + RNG.integers(2, 8))
    poss = round(float(np.clip(
        50 + (home_str - away_str) * 0.4 + RNG.normal(0, 3), 30, 70
    )), 1)

    return dict(
        home_team=home, away_team=away,
        home_goals=hg, away_goals=ag,
        home_xg=round(home_xg, 2), away_xg=round(away_xg, 2),
        home_shots=home_shots, away_shots=away_shots,
        home_shots_on_target=max(1, int(home_shots * RNG.uniform(0.30, 0.55))),
        away_shots_on_target=max(1, int(away_shots * RNG.uniform(0.25, 0.50))),
        home_possession=poss, away_possession=round(100 - poss, 1),
        home_passes=int(RNG.integers(350, 650)),
        away_passes=int(RNG.integers(300, 600)),
        home_fouls=int(RNG.integers(6, 18)),
        away_fouls=int(RNG.integers(6, 18)),
        home_yellow_cards=int(RNG.integers(0, 4)),
        away_yellow_cards=int(RNG.integers(0, 4)),
        home_red_cards=int(RNG.binomial(1, 0.05)),
        away_red_cards=int(RNG.binomial(1, 0.05)),
    )


def _matches_synthetic(players_df: pd.DataFrame) -> pd.DataFrame:
    matchweek = 0
    rows: list[dict] = []
    match_id = 1
    teams = TEAMS or _FALLBACK_TEAMS

    for rnd in range(2):
        for i, home in enumerate(teams):
            for j, away in enumerate(teams):
                if i == j:
                    continue
                if rnd == 1 and i > j:
                    continue
                matchweek += 1
                rows.append({"match_id": match_id, "matchweek": matchweek,
                              **_simulate_match(home, away, players_df)})
                match_id += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PUBLIC: generate_player_match_stats
# ---------------------------------------------------------------------------

def generate_player_match_stats(
    matches_df: pd.DataFrame,
    players_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return per-match player stat rows.

    Real mode:
      • Real top scorers get their actual season goals/assists distributed
        probabilistically across their team's matches.
      • All other attackers receive simulated shots/xG but 0 goals, so real
        scorers dominate the top-scorer chart with authentic numbers.
    Synthetic:
      • Goals distributed per-match from match totals using finishing weights.
    """
    real = _load_real_data()

    if real:
        return _player_stats_real(real["scorers"], matches_df, players_df)
    return _player_stats_synthetic(matches_df, players_df)


def _distribute(total: int, n: int, concentration: float = 0.5) -> np.ndarray:
    """Randomly distribute `total` items across `n` buckets."""
    if total == 0 or n == 0:
        return np.zeros(n, dtype=int)
    weights = RNG.dirichlet(np.ones(n) * concentration)
    return RNG.multinomial(total, weights)


def _player_stats_real(
    scorers: list[dict],
    matches_df: pd.DataFrame,
    players_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    real_player_ids: set[int] = set()

    # ---- 1. Real top scorers – distribute season totals across their matches ----
    for sc in scorers:
        team        = sc["team"]
        total_goals = sc["goals"]
        total_assts = sc["assists"]

        # Find this scorer in players_df (injected by generate_players)
        player_rows = players_df[players_df["name"] == sc["name"]]
        if player_rows.empty:
            continue
        player = player_rows.iloc[0]
        real_player_ids.add(int(player["player_id"]))

        # All matches the team played
        team_matches = matches_df[
            (matches_df["home_team"] == team) | (matches_df["away_team"] == team)
        ]
        if team_matches.empty:
            continue

        n = len(team_matches)
        total_shots = max(total_goals * 3, total_goals + int(RNG.integers(15, 40)))
        total_xg    = float(np.clip(
            total_goals * 0.82 + RNG.normal(0, 1.5), 1.0, total_goals * 1.5
        ))

        goals_dist  = _distribute(total_goals, n)
        assts_dist  = _distribute(total_assts, n)
        shots_dist  = _distribute(total_shots, n)
        xg_dist     = RNG.dirichlet(np.ones(n)) * total_xg

        for i, (_, match) in enumerate(team_matches.iterrows()):
            rows.append({
                "match_id":       match["match_id"],
                "player_id":      int(player["player_id"]),
                "name":           player["name"],
                "team":           team,
                "position":       player["position"],
                "goals":          int(goals_dist[i]),
                "assists":        int(assts_dist[i]),
                "shots":          int(shots_dist[i]),
                "xg":             round(float(xg_dist[i]), 3),
                "key_passes":     int(RNG.integers(0, 5)),
                "dribbles":       int(RNG.integers(0, 6)),
                "minutes_played": int(RNG.integers(55, 90)),
            })

    # ---- 2. Synthetic attackers – shots/xG only, 0 goals ----
    # This keeps the shot-conversion scatter interesting while real scorers
    # dominate the goals charts.
    for _, match in matches_df.iterrows():
        for side in ("home", "away"):
            team        = match[f"{side}_team"]
            total_xg    = match[f"{side}_xg"]
            total_shots = match[f"{side}_shots"]

            attackers = players_df[
                (players_df["team"] == team)
                & (players_df["position"].isin(_ATTACKER_POSITIONS))
                & (~players_df["player_id"].isin(real_player_ids))
            ]
            if attackers.empty:
                continue

            weights = attackers["finishing"].values.astype(float)
            weights /= weights.sum()

            shots_dist = RNG.multinomial(total_shots, weights)
            xg_dist    = RNG.dirichlet(weights * 5) * total_xg

            for idx, (_, player) in enumerate(attackers.iterrows()):
                rows.append({
                    "match_id":       match["match_id"],
                    "player_id":      int(player["player_id"]),
                    "name":           player["name"],
                    "team":           team,
                    "position":       player["position"],
                    "goals":          0,
                    "assists":        0,
                    "shots":          int(shots_dist[idx]),
                    "xg":             round(float(xg_dist[idx]), 3),
                    "key_passes":     int(RNG.integers(0, 5)),
                    "dribbles":       int(RNG.integers(0, 6)),
                    "minutes_played": int(RNG.integers(55, 90)),
                })

    return pd.DataFrame(rows)


def _player_stats_synthetic(
    matches_df: pd.DataFrame,
    players_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    for _, match in matches_df.iterrows():
        for side in ("home", "away"):
            team        = match[f"{side}_team"]
            total_goals = match[f"{side}_goals"]
            total_xg    = match[f"{side}_xg"]
            total_shots = match[f"{side}_shots"]

            attackers = players_df[
                (players_df["team"] == team)
                & (players_df["position"].isin(_ATTACKER_POSITIONS))
            ]
            if attackers.empty:
                continue

            weights = attackers["finishing"].values.astype(float)
            weights /= weights.sum()

            goals_dist = RNG.multinomial(total_goals, weights)
            xg_dist    = RNG.dirichlet(weights * 5) * total_xg
            shots_dist = RNG.multinomial(total_shots, weights)

            for idx, (_, player) in enumerate(attackers.iterrows()):
                rows.append({
                    "match_id":       match["match_id"],
                    "player_id":      int(player["player_id"]),
                    "name":           player["name"],
                    "team":           team,
                    "position":       player["position"],
                    "goals":          int(goals_dist[idx]),
                    "assists":        max(0, int(RNG.binomial(goals_dist[idx], 0.6))),
                    "shots":          int(shots_dist[idx]),
                    "xg":             round(float(xg_dist[idx]), 3),
                    "key_passes":     int(RNG.integers(0, 5)),
                    "dribbles":       int(RNG.integers(0, 6)),
                    "minutes_played": int(RNG.integers(60, 90)),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    players = generate_players()
    matches = generate_matches(players)
    stats   = generate_player_match_stats(matches, players)
    print(f"Teams    : {len(TEAMS)}")
    print(f"Players  : {len(players)}")
    print(f"Matches  : {len(matches)}")
    print(f"Stat rows: {len(stats)}")
