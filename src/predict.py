from __future__ import annotations

from pathlib import Path
import joblib
from .data_loading import load_matches, load_teams
from .features import build_current_team_states, feature_row_from_states


def load_artifact(model_path: str | Path):
    return joblib.load(model_path)


def predict_match(db_path: str | Path, model_path: str | Path, home_team: str, away_team: str) -> dict:
    if home_team == away_team:
        raise ValueError("Home and away teams must be different.")

    artifact = load_artifact(model_path)
    model = artifact["model"]
    teams = load_teams(db_path)
    matches = load_matches(db_path)

    name_to_id = dict(zip(teams["team_long_name"], teams["team_api_id"]))
    if home_team not in name_to_id:
        raise ValueError(f"Unknown home team: {home_team}")
    if away_team not in name_to_id:
        raise ValueError(f"Unknown away team: {away_team}")

    states = build_current_team_states(matches)
    h_id, a_id = int(name_to_id[home_team]), int(name_to_id[away_team])
    if h_id not in states or a_id not in states:
        raise ValueError("One of the teams has no match history in the database.")

    X = feature_row_from_states(states[h_id], states[a_id])
    proba = model.predict_proba(X)[0]
    probs = {cls: float(p) for cls, p in zip(model.classes_, proba)}
    prediction = model.predict(X)[0]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "prediction": prediction,
        "probabilities": probs,
        "features": X.iloc[0].to_dict(),
    }
