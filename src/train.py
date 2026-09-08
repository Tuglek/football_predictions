from __future__ import annotations

from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.tree import DecisionTreeClassifier, plot_tree

from .features import FEATURE_COLUMNS, TARGET_ORDER


def train_and_evaluate(
    dataset: pd.DataFrame,
    model_path: str | Path,
    reports_dir: str | Path,
    test_season: str = "2015/2016",
) -> dict:
    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    train = dataset[dataset["season"] != test_season].copy()
    test = dataset[dataset["season"] == test_season].copy()
    if train.empty or test.empty:
        raise ValueError(f"Could not split by test season {test_season}")

    X_train = train[FEATURE_COLUMNS]
    y_train = train["target"]
    X_test = test[FEATURE_COLUMNS]
    y_test = test["target"]

    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=6,
        min_samples_split=80,
        min_samples_leaf=35,
        class_weight=None,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba, labels=model.classes_)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=TARGET_ORDER)

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "test_season": test_season,
        "classes": list(model.classes_),
    }
    joblib.dump(artifact, model_path)

    (reports_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    pd.DataFrame(cm, index=TARGET_ORDER, columns=TARGET_ORDER).to_csv(
        reports_dir / "confusion_matrix.csv", encoding="utf-8-sig"
    )

    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(reports_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    prediction_output = test[[
        "date", "season", "league_name", "home_team", "away_team",
        "home_team_goal", "away_team_goal", "target"
    ]].copy()
    prediction_output["prediction"] = y_pred
    class_to_index = {c: i for i, c in enumerate(model.classes_)}
    for cls in TARGET_ORDER:
        prediction_output[f"p_{cls.lower()}"] = y_proba[:, class_to_index[cls]]
    prediction_output.to_csv(reports_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(18, 10))
    plot_tree(
        model,
        feature_names=FEATURE_COLUMNS,
        class_names=list(model.classes_),
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=7,
    )
    plt.title("Decision Tree: first four levels")
    plt.tight_layout()
    plt.savefig(figures_dir / "decision_tree.png", dpi=180)
    plt.close()

    top = importance.head(12).sort_values("importance")
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.title("Feature importance")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance.png", dpi=180)
    plt.close()

    metrics = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_season": test_season,
        "accuracy": float(accuracy),
        "log_loss": float(ll),
        "tree_depth": int(model.get_depth()),
        "tree_leaves": int(model.get_n_leaves()),
        "class_distribution_test": y_test.value_counts(normalize=True).round(4).to_dict(),
        "classification_report": report_dict,
    }
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics
