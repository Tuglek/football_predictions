from __future__ import annotations

import sqlite3
from pathlib import Path
import pandas as pd


def load_matches(db_path: str | Path) -> pd.DataFrame:
    """Load only the fields needed for pre-match feature engineering."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    query = """
    SELECT
        m.id,
        m.match_api_id,
        m.date,
        m.season,
        m.stage,
        m.league_id,
        l.name AS league_name,
        m.home_team_api_id,
        ht.team_long_name AS home_team,
        m.away_team_api_id,
        at.team_long_name AS away_team,
        m.home_team_goal,
        m.away_team_goal
    FROM Match AS m
    JOIN League AS l
      ON l.id = m.league_id
    JOIN Team AS ht
      ON ht.team_api_id = m.home_team_api_id
    JOIN Team AS at
      ON at.team_api_id = m.away_team_api_id
    WHERE m.home_team_goal IS NOT NULL
      AND m.away_team_goal IS NOT NULL
    ORDER BY m.date, m.id
    """

    with sqlite3.connect(db_path) as connection:
        matches = pd.read_sql_query(query, connection)

    matches["date"] = pd.to_datetime(matches["date"])
    return matches.sort_values(["date", "id"]).reset_index(drop=True)


def load_teams(db_path: str | Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as connection:
        teams = pd.read_sql_query(
            "SELECT team_api_id, team_long_name, team_short_name FROM Team ORDER BY team_long_name",
            connection,
        )
    return teams
