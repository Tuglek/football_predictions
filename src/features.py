from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

TARGET_ORDER = ["AWAY_WIN", "DRAW", "HOME_WIN"]

FEATURE_COLUMNS = [
    "home_form_points_5",
    "away_form_points_5",
    "home_win_rate_5",
    "away_win_rate_5",
    "home_goals_for_5",
    "away_goals_for_5",
    "home_goals_against_5",
    "away_goals_against_5",
    "home_season_points_per_game",
    "away_season_points_per_game",
    "home_season_win_rate",
    "away_season_win_rate",
    "home_season_goals_for",
    "away_season_goals_for",
    "home_season_goals_against",
    "away_season_goals_against",
    "home_matches_played",
    "away_matches_played",
]


@dataclass
class TeamState:
    recent: deque = field(default_factory=lambda: deque(maxlen=5))
    season: str | None = None
    season_matches: int = 0
    season_points: int = 0
    season_wins: int = 0
    season_goals_for: int = 0
    season_goals_against: int = 0

    def ensure_season(self, season: str) -> None:
        if self.season != season:
            self.season = season
            self.season_matches = 0
            self.season_points = 0
            self.season_wins = 0
            self.season_goals_for = 0
            self.season_goals_against = 0
            # Form is intentionally kept across the season boundary because
            # it represents the team's latest played matches.

    def snapshot(self) -> dict[str, float]:
        recent_games = list(self.recent)
        n_recent = len(recent_games)

        if n_recent:
            recent_points = sum(g[0] for g in recent_games)
            recent_wins = sum(g[1] for g in recent_games)
            recent_gf = sum(g[2] for g in recent_games)
            recent_ga = sum(g[3] for g in recent_games)
            form_points = recent_points / (3 * n_recent)
            win_rate = recent_wins / n_recent
            goals_for = recent_gf / n_recent
            goals_against = recent_ga / n_recent
        else:
            form_points = 0.5
            win_rate = 0.33
            goals_for = 1.3
            goals_against = 1.3

        if self.season_matches:
            ppg = self.season_points / self.season_matches
            season_win_rate = self.season_wins / self.season_matches
            season_gf = self.season_goals_for / self.season_matches
            season_ga = self.season_goals_against / self.season_matches
        else:
            ppg = 1.0
            season_win_rate = 0.33
            season_gf = 1.3
            season_ga = 1.3

        return {
            "form_points_5": form_points,
            "win_rate_5": win_rate,
            "goals_for_5": goals_for,
            "goals_against_5": goals_against,
            "season_points_per_game": ppg,
            "season_win_rate": season_win_rate,
            "season_goals_for": season_gf,
            "season_goals_against": season_ga,
            "matches_played": float(self.season_matches),
        }

    def update(self, points: int, goals_for: int, goals_against: int) -> None:
        win = int(points == 3)
        self.recent.append((points, win, goals_for, goals_against))
        self.season_matches += 1
        self.season_points += points
        self.season_wins += win
        self.season_goals_for += goals_for
        self.season_goals_against += goals_against


def _result_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "HOME_WIN"
    if home_goals < away_goals:
        return "AWAY_WIN"
    return "DRAW"


def build_feature_dataset(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Build leakage-free features chronologically.

    A row's features are calculated BEFORE the current result is added to the
    team history. Therefore no statistic contains information from the match
    that is being predicted.
    """
    states: dict[int, TeamState] = defaultdict(TeamState)
    rows: list[dict] = []

    for row in matches.sort_values(["date", "id"]).itertuples(index=False):
        home_id = int(row.home_team_api_id)
        away_id = int(row.away_team_api_id)
        season = str(row.season)

        home_state = states[home_id]
        away_state = states[away_id]
        home_state.ensure_season(season)
        away_state.ensure_season(season)

        h = home_state.snapshot()
        a = away_state.snapshot()

        record = {
            "id": row.id,
            "match_api_id": row.match_api_id,
            "date": row.date,
            "season": season,
            "stage": row.stage,
            "league_id": row.league_id,
            "league_name": row.league_name,
            "home_team_api_id": home_id,
            "home_team": row.home_team,
            "away_team_api_id": away_id,
            "away_team": row.away_team,
            "home_team_goal": int(row.home_team_goal),
            "away_team_goal": int(row.away_team_goal),
            "target": _result_label(int(row.home_team_goal), int(row.away_team_goal)),
        }

        for name, value in h.items():
            record[f"home_{name}"] = value
        for name, value in a.items():
            record[f"away_{name}"] = value

        rows.append(record)

        hg, ag = int(row.home_team_goal), int(row.away_team_goal)
        if hg > ag:
            hp, ap = 3, 0
        elif hg < ag:
            hp, ap = 0, 3
        else:
            hp, ap = 1, 1

        home_state.update(hp, hg, ag)
        away_state.update(ap, ag, hg)

    dataset = pd.DataFrame(rows)
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    return dataset


def build_current_team_states(matches: pd.DataFrame) -> dict[int, TeamState]:
    """Replay all known results and return current state for every team."""
    states: dict[int, TeamState] = defaultdict(TeamState)
    for row in matches.sort_values(["date", "id"]).itertuples(index=False):
        h_id, a_id = int(row.home_team_api_id), int(row.away_team_api_id)
        season = str(row.season)
        hs, ass = states[h_id], states[a_id]
        hs.ensure_season(season)
        ass.ensure_season(season)

        hg, ag = int(row.home_team_goal), int(row.away_team_goal)
        if hg > ag:
            hp, ap = 3, 0
        elif hg < ag:
            hp, ap = 0, 3
        else:
            hp, ap = 1, 1
        hs.update(hp, hg, ag)
        ass.update(ap, ag, hg)
    return states


def feature_row_from_states(home_state: TeamState, away_state: TeamState) -> pd.DataFrame:
    h, a = home_state.snapshot(), away_state.snapshot()
    row = {}
    for name, value in h.items():
        row[f"home_{name}"] = value
    for name, value in a.items():
        row[f"away_{name}"] = value
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
