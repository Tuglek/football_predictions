from pathlib import Path
import argparse
import pandas as pd

from src.data_loading import load_matches
from src.features import build_feature_dataset
from src.train import train_and_evaluate
from src.predict import predict_match

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "database.sqlite"
DATASET_PATH = ROOT / "data" / "prepared_matches.csv"
MODEL_PATH = ROOT / "models" / "decision_tree.joblib"
REPORTS_DIR = ROOT / "reports"


def prepare() -> pd.DataFrame:
    print("Loading matches...")
    matches = load_matches(DB_PATH)
    print(f"Loaded {len(matches):,} matches")
    print("Building leakage-free historical features...")
    dataset = build_feature_dataset(matches)
    dataset.to_csv(DATASET_PATH, index=False, encoding="utf-8-sig")
    print(f"Prepared dataset saved to: {DATASET_PATH}")
    return dataset


def train() -> None:
    if DATASET_PATH.exists():
        dataset = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    else:
        dataset = prepare()
    metrics = train_and_evaluate(dataset, MODEL_PATH, REPORTS_DIR)
    print("\nTraining finished")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Log loss: {metrics['log_loss']:.4f}")
    print(f"Tree depth: {metrics['tree_depth']}")
    print(f"Leaves: {metrics['tree_leaves']}")
    print(f"Model saved to: {MODEL_PATH}")


def predict_cli(home: str, away: str) -> None:
    result = predict_match(DB_PATH, MODEL_PATH, home, away)
    print(f"\n{home} vs {away}")
    print(f"Prediction: {result['prediction']}")
    p = result["probabilities"]
    print(f"Home win: {p.get('HOME_WIN', 0):.2%}")
    print(f"Draw:     {p.get('DRAW', 0):.2%}")
    print(f"Away win: {p.get('AWAY_WIN', 0):.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="European Soccer Decision Tree project")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("train")
    pred = sub.add_parser("predict")
    pred.add_argument("--home", required=True)
    pred.add_argument("--away", required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        prepare()
    elif args.command == "train":
        train()
    elif args.command == "predict":
        predict_cli(args.home, args.away)


if __name__ == "__main__":
    main()
