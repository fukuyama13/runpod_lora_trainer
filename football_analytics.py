"""
Football Analytics Script
=========================
Analyses a Premier League season and produces:
  1. League standings table
  2. Top scorers / assisters
  3. xG vs actual goals (over/under-performance)
  4. Team defensive strength
  5. Player radar charts
  6. Match outcome heatmap
  7. xG rolling trend per team (top 6)
  8. Shot conversion efficiency scatter

Data source (in priority order):
  1. Real data from football-data.org  (set FOOTBALL_API_KEY in .env)
  2. Fully synthetic simulation        (automatic fallback)

Run:
    pip install -r requirements.txt
    python football_analytics.py
"""

from __future__ import annotations

import argparse
from typing import Optional
import json
import math
import sys
import io
import warnings
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
from scipy import stats
from tabulate import tabulate

import sample_data

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("analytics_output")
OUTPUT_DIR.mkdir(exist_ok=True)

STYLE = "seaborn-v0_8-darkgrid"
PALETTE = "tab20"
TOP_N_SCORERS = 10
# Populated dynamically from standings after data loads
TOP_6: list[str] = []

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "figure.dpi":   120,
})


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global TOP_6
    print("Loading season data …")
    players      = sample_data.generate_players()
    matches      = sample_data.generate_matches(players)
    player_stats = sample_data.generate_player_match_stats(matches, players)

    # Derive TOP_6 from the strongest teams in this dataset
    team_pts: dict[str, int] = {}
    for _, m in matches.iterrows():
        for side, opp in (("home", "away"), ("away", "home")):
            t  = m[f"{side}_team"]
            gf = m[f"{side}_goals"]
            ga = m[f"{opp}_goals"]
            pts = 3 if gf > ga else (1 if gf == ga else 0)
            team_pts[t] = team_pts.get(t, 0) + pts
    TOP_6 = sorted(team_pts, key=lambda t: team_pts[t], reverse=True)[:6]

    source = "real API data" if sample_data._real_cache else "synthetic data"
    print(f"  {len(players)} players | {len(matches)} matches | "
          f"{len(player_stats)} stat rows  [{source}]\n")
    return players, matches, player_stats


# ---------------------------------------------------------------------------
# 1. League standings
# ---------------------------------------------------------------------------

def build_standings(matches: pd.DataFrame) -> pd.DataFrame:
    all_teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    records: dict[str, dict] = {
        t: dict(P=0, W=0, D=0, L=0, GF=0, GA=0, GD=0, Pts=0, xGF=0.0, xGA=0.0)
        for t in all_teams
    }

    for _, m in matches.iterrows():
        ht, at = m.home_team, m.away_team
        hg, ag = m.home_goals, m.away_goals

        for team, gf, ga, xgf, xga in [
            (ht, hg, ag, m.home_xg, m.away_xg),
            (at, ag, hg, m.away_xg, m.home_xg),
        ]:
            r = records[team]
            r["P"]   += 1
            r["GF"]  += gf
            r["GA"]  += ga
            r["GD"]  += gf - ga
            r["xGF"] += xgf
            r["xGA"] += xga
            if gf > ga:
                r["W"] += 1; r["Pts"] += 3
            elif gf == ga:
                r["D"] += 1; r["Pts"] += 1
            else:
                r["L"] += 1

    df = pd.DataFrame(records).T.reset_index().rename(columns={"index": "Team"})
    df = df.sort_values(["Pts", "GD", "GF"], ascending=False).reset_index(drop=True)
    df.index += 1
    df["xGD"] = (df["xGF"] - df["xGA"]).round(1)
    df["xGF"] = df["xGF"].round(1)
    df["xGA"] = df["xGA"].round(1)
    return df


def print_standings(standings: pd.DataFrame) -> None:
    cols = ["Team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts", "xGF", "xGA", "xGD"]
    print("=" * 70)
    print("  PREMIER LEAGUE STANDINGS")
    print("=" * 70)
    print(tabulate(standings[cols], headers="keys", tablefmt="simple"))
    print()


# ---------------------------------------------------------------------------
# 2. Player aggregates
# ---------------------------------------------------------------------------

def aggregate_player_stats(player_stats: pd.DataFrame) -> pd.DataFrame:
    agg = (
        player_stats
        .groupby(["player_id", "name", "team", "position"], as_index=False)
        .agg(
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            shots=("shots", "sum"),
            xg=("xg", "sum"),
            key_passes=("key_passes", "sum"),
            dribbles=("dribbles", "sum"),
            minutes=("minutes_played", "sum"),
            appearances=("match_id", "count"),
        )
    )
    agg["goal_contributions"] = agg["goals"] + agg["assists"]
    agg["xg_overperformance"] = (agg["goals"] - agg["xg"]).round(2)
    agg["shot_conversion"]    = (agg["goals"] / agg["shots"].replace(0, np.nan) * 100).round(1)
    agg["goals_per_90"]       = (agg["goals"] / agg["minutes"] * 90).round(2)
    return agg.sort_values("goals", ascending=False).reset_index(drop=True)


def print_top_scorers(agg: pd.DataFrame, n: int = TOP_N_SCORERS) -> None:
    cols = ["name", "team", "position", "goals", "assists",
            "goal_contributions", "xg", "xg_overperformance",
            "shot_conversion", "goals_per_90"]
    print("=" * 70)
    print(f"  TOP {n} SCORERS")
    print("=" * 70)
    print(tabulate(agg[cols].head(n), headers="keys", tablefmt="simple",
                   floatfmt=".2f"))
    print()


# ---------------------------------------------------------------------------
# 3. Team defensive stats
# ---------------------------------------------------------------------------

def team_defensive_stats(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[["home_team", "away_goals", "away_shots_on_target", "away_xg"]].copy()
    home.columns = ["team", "goals_conceded", "shots_on_target_conceded", "xga"]

    away = matches[["away_team", "home_goals", "home_shots_on_target", "home_xg"]].copy()
    away.columns = ["team", "goals_conceded", "shots_on_target_conceded", "xga"]

    combined = pd.concat([home, away])
    return (
        combined
        .groupby("team", as_index=False)
        .agg(
            goals_conceded=("goals_conceded", "sum"),
            shots_conceded=("shots_on_target_conceded", "sum"),
            xga=("xga", "sum"),
        )
        .assign(xga=lambda d: d["xga"].round(1))
        .sort_values("goals_conceded")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Plot 1 – League table bar chart
# ---------------------------------------------------------------------------

def plot_standings(standings: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette(PALETTE, len(standings))
    bars = ax.barh(standings["Team"][::-1], standings["Pts"][::-1], color=colors[::-1])
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.set_xlabel("Points")
    ax.set_title("Premier League Final Standings")
    ax.axvline(standings["Pts"].iloc[3], color="gold", lw=1.5, ls="--",
               label="UCL boundary")
    ax.axvline(standings["Pts"].iloc[16], color="red", lw=1.5, ls="--",
               label="Relegation boundary")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "01_standings.png")


# ---------------------------------------------------------------------------
# Plot 2 – Top scorers
# ---------------------------------------------------------------------------

def plot_top_scorers(agg: pd.DataFrame, n: int = TOP_N_SCORERS) -> None:
    top = agg.head(n).copy()
    x = np.arange(n)
    width = 0.4

    fig, ax = plt.subplots(figsize=(11, 5))
    bars_g = ax.bar(x - width / 2, top["goals"],   width, label="Goals",   color="#2196F3")
    bars_a = ax.bar(x + width / 2, top["assists"], width, label="Assists", color="#4CAF50")

    for bar in bars_g:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)
    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['name']}\n({r['team']})" for _, r in top.iterrows()],
        fontsize=8, rotation=20, ha="right",
    )
    ax.set_ylabel("Count")
    ax.set_title(f"Top {n} Scorers – Goals & Assists")
    ax.legend()
    fig.tight_layout()
    _save(fig, "02_top_scorers.png")


# ---------------------------------------------------------------------------
# Plot 3 – xG vs actual goals (team level)
# ---------------------------------------------------------------------------

def plot_xg_vs_goals(standings: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(standings["xGF"], standings["GF"],
               c=standings["Pts"], cmap="RdYlGn", s=120, zorder=3)

    lim_lo = min(standings["xGF"].min(), standings["GF"].min()) - 2
    lim_hi = max(standings["xGF"].max(), standings["GF"].max()) + 2
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, label="Expected = Actual")

    for _, row in standings.iterrows():
        ax.annotate(row["Team"], (row["xGF"], row["GF"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7)

    sm = plt.cm.ScalarMappable(cmap="RdYlGn",
                               norm=plt.Normalize(standings["Pts"].min(),
                                                  standings["Pts"].max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Points")

    ax.set_xlabel("Expected Goals For (xGF)")
    ax.set_ylabel("Actual Goals For")
    ax.set_title("xG vs Actual Goals – Team Performance")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "03_xg_vs_goals.png")


# ---------------------------------------------------------------------------
# Plot 4 – Defensive strength
# ---------------------------------------------------------------------------

def plot_defensive_strength(def_stats: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("coolwarm", len(def_stats))
    bars = ax.barh(def_stats["team"], def_stats["goals_conceded"], color=palette)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)

    ax2 = ax.twiny()
    ax2.scatter(def_stats["xga"], def_stats["team"],
                color="navy", zorder=5, s=50, label="xGA")
    ax2.set_xlabel("xGA", color="navy")
    ax2.tick_params(axis="x", colors="navy")

    ax.set_xlabel("Goals Conceded")
    ax.set_title("Defensive Strength – Goals Conceded vs xGA")
    ax2.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _save(fig, "04_defensive_strength.png")


# ---------------------------------------------------------------------------
# Plot 5 – Player radar chart (top 3 attackers)
# ---------------------------------------------------------------------------

def _radar(ax: plt.Axes, values: list[float], labels: list[str],
           color: str, label: str) -> None:
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles      = angles + [angles[0]]
    ax.plot(angles, values_plot, color=color, linewidth=2, label=label)
    ax.fill(angles, values_plot, color=color, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_yticklabels([])
    ax.set_ylim(0, 100)


def plot_player_radar(players: pd.DataFrame, agg: pd.DataFrame) -> None:
    top3 = agg.head(3)
    attrs = ["finishing", "passing", "pace", "physicality", "defending"]
    colors = ["#E63946", "#2196F3", "#4CAF50"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             subplot_kw=dict(polar=True))

    for ax, (_, player_row), color in zip(axes, top3.iterrows(), colors):
        pid = player_row["player_id"]
        p   = players[players["player_id"] == pid].iloc[0]
        vals = [p[a] for a in attrs]
        _radar(ax, vals, attrs, color, p["name"])
        ax.set_title(f"{p['name']}\n{p['team']}", size=10, pad=15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)

    fig.suptitle("Player Radar – Top 3 Scorers", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, "05_player_radar.png")


# ---------------------------------------------------------------------------
# Plot 6 – Match outcome heatmap (home vs away wins/draws)
# ---------------------------------------------------------------------------

def plot_outcome_heatmap(matches: pd.DataFrame) -> None:
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    matrix = pd.DataFrame(np.nan, index=teams, columns=teams)

    for _, m in matches.iterrows():
        hg, ag = m.home_goals, m.away_goals
        result = 3 if hg > ag else (1 if hg == ag else 0)
        matrix.loc[m.home_team, m.away_team] = result

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(10, 145, as_cmap=True)
    mask = matrix.isna()
    sns.heatmap(
        matrix.fillna(0), ax=ax, cmap=cmap, vmin=0, vmax=3,
        linewidths=0.4, linecolor="white", mask=mask,
        cbar_kws={"label": "0=Loss  1=Draw  3=Win (home perspective)"},
        annot=True, fmt=".0f", annot_kws={"size": 6},
    )
    ax.set_title("Match Outcome Heatmap (home team = row, away team = col)")
    ax.set_xlabel("Away Team")
    ax.set_ylabel("Home Team")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    fig.tight_layout()
    _save(fig, "06_outcome_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 7 – xG rolling trend for Top 6
# ---------------------------------------------------------------------------

def plot_xg_trend(matches: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("tab10", len(TOP_6))

    for team, color in zip(TOP_6, colors):
        home = matches[matches["home_team"] == team][["matchweek", "home_xg"]].rename(
            columns={"home_xg": "xg"})
        away = matches[matches["away_team"] == team][["matchweek", "away_xg"]].rename(
            columns={"away_xg": "xg"})
        team_xg = pd.concat([home, away]).sort_values("matchweek")
        rolling = team_xg["xg"].rolling(5, min_periods=1).mean()
        ax.plot(team_xg["matchweek"], rolling, label=team, color=color, lw=2)

    ax.set_xlabel("Matchweek")
    ax.set_ylabel("xG (5-match rolling avg)")
    ax.set_title("xG Rolling Trend – Top 6 Teams")
    ax.legend(fontsize=9, loc="upper left")
    ax.axhline(1.5, color="grey", ls="--", lw=1, alpha=0.6, label="Avg threshold")
    fig.tight_layout()
    _save(fig, "07_xg_trend.png")


# ---------------------------------------------------------------------------
# Plot 8 – Shot conversion efficiency scatter
# ---------------------------------------------------------------------------

def plot_shot_conversion(agg: pd.DataFrame) -> None:
    df = agg[(agg["shots"] >= 5) & agg["shot_conversion"].notna()].copy()

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        df["shots"], df["goals"],
        c=df["xg_overperformance"],
        cmap="RdYlGn", s=df["shot_conversion"] * 3 + 20,
        alpha=0.75, edgecolors="white", linewidths=0.5,
    )
    plt.colorbar(scatter, ax=ax, label="xG Over-performance")

    # Trend line
    slope, intercept, r, *_ = stats.linregress(df["shots"], df["goals"])
    x_line = np.linspace(df["shots"].min(), df["shots"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "k--", lw=1.5,
            label=f"Trend  R²={r**2:.2f}")

    # Label top performers
    for _, row in df.nlargest(5, "goals").iterrows():
        ax.annotate(row["name"].split()[-1],
                    (row["shots"], row["goals"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)

    ax.set_xlabel("Total Shots")
    ax.set_ylabel("Goals Scored")
    ax.set_title("Shot Conversion Efficiency\n(bubble size = conversion %, colour = xG over-performance)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "08_shot_conversion.png")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(standings: pd.DataFrame, agg: pd.DataFrame,
                  def_stats: pd.DataFrame) -> None:
    champion   = standings.iloc[0]
    top_scorer = agg.iloc[0]
    best_def   = def_stats.iloc[0]

    overperf = agg.loc[agg["xg_overperformance"].idxmax()]
    underperf = agg.loc[agg["xg_overperformance"].idxmin()]

    print("=" * 70)
    print("  SEASON SUMMARY")
    print("=" * 70)
    print(f"  Champion        : {champion['Team']}  ({int(champion['Pts'])} pts)")
    print(f"  Top Scorer      : {top_scorer['name']} ({top_scorer['team']}) "
          f"– {int(top_scorer['goals'])} goals")
    print(f"  Best Defence    : {best_def['team']} "
          f"– {int(best_def['goals_conceded'])} goals conceded")
    print(f"  xG Overperformer: {overperf['name']} ({overperf['team']}) "
          f"+{overperf['xg_overperformance']:.2f} vs xG")
    print(f"  xG Underperformer: {underperf['name']} ({underperf['team']}) "
          f"{underperf['xg_overperformance']:.2f} vs xG")
    print()

    relegated = standings.tail(3)["Team"].tolist()
    ucl = standings.head(4)["Team"].tolist()
    print(f"  UCL spots       : {', '.join(ucl)}")
    print(f"  Relegated       : {', '.join(relegated)}")
    print()


# ---------------------------------------------------------------------------
# Dashboard data export
# ---------------------------------------------------------------------------

def _safe_float(v) -> float | None:
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 2)
    except (TypeError, ValueError):
        return None


def _safe_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _build_statsbomb_section(match_id: Optional[int]) -> dict:
    """Return the StatsBomb sub-section for data.json."""
    try:
        import statsbomb_loader as _sb
        import tools as _tools
    except ImportError as exc:
        return {"available": False, "message": f"Import error: {exc}"}

    if not _sb.is_available():
        return {
            "available": False,
            "message":   (
                "Clone the StatsBomb open-data repo first:\n"
                "  git clone https://github.com/statsbomb/open-data.git statsbomb_data\n"
                "Then re-run:  python football_analytics.py --export"
            ),
        }

    print("  StatsBomb: loading catalogue …", end="", flush=True)
    catalog = _tools.get_match_catalog()
    print(f" {sum(len(c['matches']) for c in catalog)} matches across "
          f"{len(catalog)} competition-seasons")

    active_match: Optional[dict] = None

    # Pick which match to load event data for
    target_id = match_id
    if target_id is None and catalog:
        # Auto-select the first match of the first competition as a demo
        target_id = catalog[0]["matches"][0]["match_id"] if catalog[0]["matches"] else None

    if target_id is not None:
        print(f"  StatsBomb: loading events for match {target_id} …", end="", flush=True)
        active_match = _tools.build_active_match_payload(target_id)
        if "error" not in active_match:
            # Inject competition/season metadata from catalogue
            for comp in catalog:
                for m in comp["matches"]:
                    if m["match_id"] == target_id:
                        active_match["match_date"] = m["match_date"]
                        active_match["competition"] = comp["competition"]
                        active_match["season"]      = comp["season"]
                        active_match["country"]     = comp["country"]
                        break
            n_shots  = len(active_match.get("shots", []))
            n_events = sum(len(v) for v in active_match.get("heatmaps", {}).values())
            print(f" {n_shots} shots, {n_events} heatmap touches")
        else:
            print(f" ERROR: {active_match['error']}")

    return {
        "available":    True,
        "catalog":      catalog,
        "active_match": active_match,
    }


def export_data(
    standings: pd.DataFrame,
    agg: pd.DataFrame,
    def_stats: pd.DataFrame,
    matches: pd.DataFrame,
    players_df: pd.DataFrame,
    statsbomb_match_id: Optional[int] = None,
) -> None:
    """Write analytics_output/data.json and inject inline into dashboard.html."""

    # -- Form: last 5 results per team in chronological order ---------------
    def get_form(team: str, n: int = 5) -> list[str]:
        tm = (
            matches[
                (matches["home_team"] == team) | (matches["away_team"] == team)
            ]
            .sort_values(["matchweek", "match_id"], ascending=False)
            .head(n)
        )
        form: list[str] = []
        for _, m in tm.iterrows():
            gf = m["home_goals"] if m["home_team"] == team else m["away_goals"]
            ga = m["away_goals"] if m["home_team"] == team else m["home_goals"]
            form.append("W" if gf > ga else ("D" if gf == ga else "L"))
        return list(reversed(form))   # oldest → newest

    # -- Standings -----------------------------------------------------------
    standings_list = []
    for pos, (_, row) in enumerate(standings.iterrows(), 1):
        team = row["Team"]
        gd   = _safe_int(row["GD"])
        standings_list.append({
            "position": pos,
            "team":  team,
            "P":     _safe_int(row["P"]),
            "W":     _safe_int(row["W"]),
            "D":     _safe_int(row["D"]),
            "L":     _safe_int(row["L"]),
            "GF":    _safe_int(row["GF"]),
            "GA":    _safe_int(row["GA"]),
            "GD":    gd,
            "GD_str": f"+{gd}" if gd >= 0 else str(gd),
            "Pts":   _safe_int(row["Pts"]),
            "xGF":   _safe_float(row["xGF"]),
            "xGA":   _safe_float(row["xGA"]),
            "xGD":   _safe_float(row["xGD"]),
            "form":  get_form(team),
        })

    # -- Top scorers ---------------------------------------------------------
    scorers_list = []
    for _, row in agg.head(15).iterrows():
        scorers_list.append({
            "name":               row["name"],
            "team":               row["team"],
            "position":           row["position"],
            "goals":              _safe_int(row["goals"]),
            "assists":            _safe_int(row["assists"]),
            "xg":                 _safe_float(row["xg"]) or 0.0,
            "xg_overperformance": _safe_float(row["xg_overperformance"]) or 0.0,
            "shot_conversion":    _safe_float(row["shot_conversion"]) or 0.0,
            "goals_per_90":       _safe_float(row["goals_per_90"]) or 0.0,
        })

    # -- xG analysis (team level) -------------------------------------------
    xg_list = [
        {
            "team": row["Team"],
            "xGF":  _safe_float(row["xGF"]),
            "GF":   _safe_int(row["GF"]),
            "Pts":  _safe_int(row["Pts"]),
        }
        for _, row in standings.iterrows()
    ]

    # -- Matches (flat list for H2H + team pages) ----------------------------
    matches_list = [
        {
            "mw": int(m["matchweek"]),
            "ht": m["home_team"],
            "at": m["away_team"],
            "hg": int(m["home_goals"]),
            "ag": int(m["away_goals"]),
            "hx": _safe_float(m["home_xg"]) or 0.0,
            "ax": _safe_float(m["away_xg"]) or 0.0,
        }
        for _, m in matches.sort_values("matchweek").iterrows()
    ]

    # -- Team possession averages (estimated from pass/possession columns) ---
    home_p = matches[["home_team", "home_possession"]].rename(
        columns={"home_team": "team", "home_possession": "poss"})
    away_p = matches[["away_team", "away_possession"]].rename(
        columns={"away_team": "team", "away_possession": "poss"})
    avg_poss = (
        pd.concat([home_p, away_p])
        .groupby("team")["poss"].mean().round(1)
    )

    # Inject possession into standings_list
    for entry in standings_list:
        entry["poss"] = float(avg_poss.get(entry["team"], 50.0))

    # -- Players (for profiles tab) ------------------------------------------
    # Merge season-aggregate stats with per-player attributes
    players_list = []
    for _, row in agg.iterrows():
        pid   = int(row["player_id"])
        p_row = players_df[players_df["player_id"] == pid]
        attrs = p_row.iloc[0] if not p_row.empty else None

        def _a(col: str, default=0):
            return _safe_int(attrs[col]) if attrs is not None and col in attrs else default

        players_list.append({
            "id":          pid,
            "name":        row["name"],
            "team":        row["team"],
            "pos":         row["position"],
            "nat":         str(attrs["nationality"]) if attrs is not None else "—",
            "age":         _a("age"),
            "fin":         _a("finishing"),
            "pas":         _a("passing"),
            "pac":         _a("pace"),
            "phy":         _a("physicality"),
            "def":         _a("defending"),
            "goals":       _safe_int(row["goals"]),
            "assists":     _safe_int(row["assists"]),
            "shots":       _safe_int(row["shots"]),
            "xg":          _safe_float(row["xg"]) or 0.0,
            "xg_op":       _safe_float(row["xg_overperformance"]) or 0.0,
            "conv":        _safe_float(row["shot_conversion"]) or 0.0,
            "g90":         _safe_float(row["goals_per_90"]) or 0.0,
            "apps":        _safe_int(row["appearances"]),
            "kp":          _safe_int(row["key_passes"]),
        })

    # -- Summary -------------------------------------------------------------
    champion   = standings.iloc[0]
    top_scorer = agg.iloc[0]
    best_def   = def_stats.iloc[0]
    overperf   = agg.loc[agg["xg_overperformance"].idxmax()]

    summary = {
        "champion":               champion["Team"],
        "champion_pts":           _safe_int(champion["Pts"]),
        "champion_gd":            _safe_int(champion["GD"]),
        "champion_gf":            _safe_int(champion["GF"]),
        "top_scorer_name":        top_scorer["name"],
        "top_scorer_team":        top_scorer["team"],
        "top_scorer_goals":       _safe_int(top_scorer["goals"]),
        "top_scorer_assists":     _safe_int(top_scorer["assists"]),
        "best_defence_team":      best_def["team"],
        "best_defence_goals":     _safe_int(best_def["goals_conceded"]),
        "xg_overperformer_name":  overperf["name"],
        "xg_overperformer_team":  overperf["team"],
        "xg_overperformance":     _safe_float(overperf["xg_overperformance"]) or 0.0,
        "ucl_teams":              standings["Team"].head(4).tolist(),
        "europa_team":            standings["Team"].iloc[4] if len(standings) > 4 else "",
        "conference_team":        standings["Team"].iloc[5] if len(standings) > 5 else "",
        "relegated_teams":        standings["Team"].tail(3).tolist(),
    }

    import re as _re

    is_real = bool(sample_data._real_cache)
    payload = {
        "generated_at": datetime.now().strftime("%d %b %Y, %H:%M"),
        "data_source":  "Real API data" if is_real else "Synthetic simulation",
        "standings":    standings_list,
        "top_scorers":  scorers_list,
        "xg_analysis":  xg_list,
        "matches":      matches_list,
        "players":      players_list,
        "summary":      summary,
        "statsbomb":    _build_statsbomb_section(statsbomb_match_id),
    }

    json_str = json.dumps(payload, indent=2, ensure_ascii=False)

    # 1. Write analytics_output/data.json  (pure JSON for inspection)
    json_path = OUTPUT_DIR / "data.json"
    json_path.write_text(json_str, encoding="utf-8")
    print(f"  data.json  → {json_path}  [{payload['data_source']}]")

    # 2. Inject data inline into dashboard.html so it works from file://
    #    (fetch() is blocked by Chrome CORS when opening a local HTML file,
    #     but an inline <script> tag has no such restriction).
    html_path = Path("dashboard.html")
    if not html_path.exists():
        print("  WARNING: dashboard.html not found — skipping inline injection")
        return

    html = html_path.read_text(encoding="utf-8")
    inline = (
        "<!-- ##DATA_SCRIPT_START## -->\n"
        f"<script>/* {payload['data_source']} · {payload['generated_at']} */\n"
        f"window.FOOTBALL_DATA={json_str};\n"
        "</script>\n"
        "<!-- ##DATA_SCRIPT_END## -->"
    )
    pattern = r"<!-- ##DATA_SCRIPT_START## -->.*?<!-- ##DATA_SCRIPT_END## -->"
    if _re.search(pattern, html, _re.DOTALL):
        html = _re.sub(pattern, inline, html, flags=_re.DOTALL)
    else:
        # Fallback: insert before </head> if markers were never placed
        html = html.replace("</head>", inline + "\n</head>", 1)

    html_path.write_text(html, encoding="utf-8")
    print(f"  dashboard  → {html_path}  (data embedded inline — open directly in browser)")

    # Warn if still on synthetic data
    if not is_real:
        print()
        print("  *** SHOWING SYNTHETIC DATA ***")
        print("  To load REAL Premier League data:")
        print("  1. Get a free API key at https://www.football-data.org/client/register")
        print("  2. Open .env and set FOOTBALL_API_KEY=<your_key>")
        print("  3. Delete api_cache/ (if it exists) to force a fresh fetch")
        print("  4. Re-run: python football_analytics.py --export")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Premier League Football Analytics")
    parser.add_argument(
        "--export", action="store_true",
        help="Write analytics_output/data.json and embed data inline in dashboard.html",
    )
    parser.add_argument(
        "--statsbomb-match", type=int, default=None, metavar="MATCH_ID",
        help="StatsBomb match ID to include full event data in the export",
    )
    args = parser.parse_args()

    players, matches, player_stats = load_data()

    standings  = build_standings(matches)
    agg        = aggregate_player_stats(player_stats)
    def_stats  = team_defensive_stats(matches)

    print_standings(standings)
    print_top_scorers(agg)
    print_summary(standings, agg, def_stats)

    print("Generating charts …")
    plot_standings(standings)
    plot_top_scorers(agg)
    plot_xg_vs_goals(standings)
    plot_defensive_strength(def_stats)
    plot_player_radar(players, agg)
    plot_outcome_heatmap(matches)
    plot_xg_trend(matches)
    plot_shot_conversion(agg)

    print(f"\nAll charts saved to ./{OUTPUT_DIR}/")

    if args.export:
        print("\nExporting dashboard data …")
        export_data(standings, agg, def_stats, matches, players,
                    statsbomb_match_id=args.statsbomb_match)


if __name__ == "__main__":
    main()
