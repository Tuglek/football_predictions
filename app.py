from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from src.data_loading import load_matches, load_teams
from src.features import build_current_team_states, feature_row_from_states

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "database.sqlite"
MODEL_PATH = ROOT / "models" / "decision_tree.joblib"
PREDICTIONS_PATH = ROOT / "reports" / "test_predictions.csv"

st.set_page_config(page_title="Прогноз футбольного матча", page_icon="⚽", layout="centered")
st.title("⚽ Прогноз результата футбольного матча")
st.caption("Решающее дерево, обученное на European Soccer Database")

if not MODEL_PATH.exists():
    st.error("Модель не найдена. Сначала выполните: python main.py train")
    st.stop()

@st.cache_resource
def load_runtime():
    artifact = joblib.load(MODEL_PATH)
    teams = load_teams(DB_PATH)
    matches = load_matches(DB_PATH)
    states = build_current_team_states(matches)
    return artifact, teams, states

@st.cache_data
def load_historical_predictions():
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])
    return df.sort_values("date", ascending=False).reset_index(drop=True)

artifact, teams, states = load_runtime()
model = artifact["model"]
valid_teams = teams[teams["team_api_id"].isin(states.keys())].copy()
team_names = valid_teams["team_long_name"].sort_values().tolist()
name_to_id = dict(zip(valid_teams["team_long_name"], valid_teams["team_api_id"]))

labels = {
    "HOME_WIN": "Победа хозяев",
    "DRAW": "Ничья",
    "AWAY_WIN": "Победа гостей",
}


tab_predict, tab_check = st.tabs(["🔮 Новый прогноз", "✅ Проверка прогноза"])

with tab_predict:
    st.subheader("Предматчевый прогноз")
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Хозяева", team_names, index=0, key="home_predict")
    with col2:
        default_away = 1 if len(team_names) > 1 else 0
        away_team = st.selectbox("Гости", team_names, index=default_away, key="away_predict")

    if st.button("Рассчитать вероятность", type="primary", use_container_width=True):
        if home_team == away_team:
            st.warning("Выберите две разные команды.")
        else:
            h_id = int(name_to_id[home_team])
            a_id = int(name_to_id[away_team])
            X = feature_row_from_states(states[h_id], states[a_id])
            probabilities = dict(zip(model.classes_, model.predict_proba(X)[0]))
            prediction = model.predict(X)[0]

            st.subheader("Прогноз")
            st.success(labels[prediction])

            c1, c2, c3 = st.columns(3)
            c1.metric("Победа хозяев", f"{probabilities.get('HOME_WIN', 0):.1%}")
            c2.metric("Ничья", f"{probabilities.get('DRAW', 0):.1%}")
            c3.metric("Победа гостей", f"{probabilities.get('AWAY_WIN', 0):.1%}")

            chart = pd.DataFrame({
                "Исход": ["Победа хозяев", "Ничья", "Победа гостей"],
                "Вероятность": [
                    probabilities.get("HOME_WIN", 0),
                    probabilities.get("DRAW", 0),
                    probabilities.get("AWAY_WIN", 0),
                ],
            }).set_index("Исход")
            st.bar_chart(chart)

            st.info(
                "Это предматчевый прогноз. Его нельзя автоматически считать правильным "
                "до завершения матча. Для проверки фактического результата используйте вкладку «Проверка прогноза»."
            )

            with st.expander("Признаки, переданные модели"):
                st.dataframe(X.T.rename(columns={0: "Значение"}), use_container_width=True)

with tab_check:
    st.subheader("Проверка исторических прогнозов")
    st.caption(
        "Здесь используются прогнозы на тестовой выборке сезона 2015/2016. "
        "Они были рассчитаны моделью до того, как фактический результат матча был известен."
    )

    history = load_historical_predictions()
    if history.empty:
        st.warning("Файл reports/test_predictions.csv не найден. Сначала выполните: python main.py train")
    else:
        selected_date = st.date_input(
            "Дата матча",
            value=history.iloc[0]["date"].date(),
            min_value=history["date"].min().date(),
            max_value=history["date"].max().date(),
            key="check_date",
        )

        day_matches = history[history["date"].dt.date == selected_date].copy()
        if day_matches.empty:
            st.info("На выбранную дату матчей в тестовой выборке нет.")
        else:
            day_matches["match"] = day_matches["home_team"] + " — " + day_matches["away_team"]
            selected_match = st.selectbox(
                "Матч",
                day_matches.index,
                format_func=lambda idx: day_matches.loc[idx, "match"],
                key="historical_match",
            )

            row = day_matches.loc[selected_match]
            predicted = row["prediction"]
            actual = row["target"]
            correct = predicted == actual

            st.markdown(f"### {row['home_team']} — {row['away_team']}")
            st.write(f"Дата: **{row['date'].strftime('%d.%m.%Y')}**")

            c1, c2, c3 = st.columns(3)
            c1.metric("Победа хозяев", f"{row['p_home_win']:.1%}")
            c2.metric("Ничья", f"{row['p_draw']:.1%}")
            c3.metric("Победа гостей", f"{row['p_away_win']:.1%}")

            st.write(f"**Предполагаемый исход:** {labels[predicted]}")
            st.write(
                f"**Фактический счёт:** {int(row['home_team_goal'])} : {int(row['away_team_goal'])}"
            )
            st.write(f"**Фактический исход:** {labels[actual]}")

            if correct:
                st.success("✅ ПРОГНОЗ ВЕРНЫЙ — предполагаемый и фактический исход совпали.")
            else:
                st.error("❌ ПРОГНОЗ НЕВЕРНЫЙ — предполагаемый и фактический исход различаются.")

            with st.expander("Подробности проверки"):
                comparison = pd.DataFrame({
                    "Показатель": ["Прогноз модели", "Фактический исход", "Результат проверки"],
                    "Значение": [
                        labels[predicted],
                        labels[actual],
                        "Верный" if correct else "Неверный",
                    ],
                })
                st.table(comparison)

        st.divider()
        st.subheader("Общая статистика тестовой выборки")

        # Проверяем наличие фактического результата
        if "target" not in history.columns:
            if "result" in history.columns:
                history["target"] = history["result"]
            else:
                history["target"] = None

        # Учитываем только матчи, для которых известен фактический результат
        history_with_target = history.dropna(subset=["target"]).copy()

        if history_with_target.empty:
            st.warning("В истории нет фактических результатов матчей.")
        else:
            total = len(history_with_target)

            history_with_target["correct"] = (
                    history_with_target["prediction"]
                    == history_with_target["target"]
            )

            correct_total = int(history_with_target["correct"].sum())
            incorrect_total = total - correct_total
            accuracy = correct_total / total if total else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Всего матчей", f"{total:,}")
            c2.metric("Правильных прогнозов", f"{correct_total:,}")
            c3.metric("Точность", f"{accuracy:.1%}")

            st.write(
                f"Неправильных прогнозов: **{incorrect_total:,}**"
            )

            # Статистика по каждому типу исхода
            by_class = history_with_target.groupby("target").apply(
                lambda x: pd.Series({
                    "Матчей": len(x),
                    "Правильных": int(x["correct"].sum()),
                    "Точность": x["correct"].mean()
                }),
                include_groups=False
            )

            by_class.index = [
                labels.get(i, i)
                for i in by_class.index
            ]

            by_class["Точность"] = by_class["Точность"].map(
                lambda x: f"{x:.1%}"
            )

            st.dataframe(
                by_class,
                use_container_width=True
            )