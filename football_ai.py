import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
import datetime

# ===========================
# LEAGUE MAPPING (SofaScore IDs)
# ===========================
LEAGUES = {
    "All Matches Today 🌍": None,
    "Premier League": {"id": 17, "season": 41886},
    "La Liga": {"id": 8, "season": 41868},
    "Serie A": {"id": 23, "season": 41895},
    "Bundesliga": {"id": 35, "season": 41919},
    "Champions League": {"id": 7, "season": 41864}
}

# ===========================
# SOFASCORE FUNCTIONS
# ===========================
def get_upcoming_matches(league=None):
    try:
        if league is None:
            today = datetime.date.today().strftime("%Y-%m-%d")
            url = f"https://api.sofascore.com/api/v1/sport/football/scheduled-events/{today}"
        else:
            url = f"https://api.sofascore.com/api/v1/unique-tournament/{league['id']}/season/{league['season']}/events"
        resp = requests.get(url).json()
        matches = []
        for m in resp.get("events", []):
            home = m["homeTeam"]["name"]
            away = m["awayTeam"]["name"]
            tournament = m.get("tournament", {}).get("name", "")
            matches.append({
                "label": f"{home} vs {away} ({tournament})",
                "home": home,
                "away": away,
                "id": m["id"]
            })
        return matches
    except Exception as e:
        return []

def get_sofascore_stats(match_id):
    url = f"https://api.sofascore.com/api/v1/event/{match_id}/statistics"
    resp = requests.get(url).json()
    stats = {}
    for group in resp.get("statistics", []):
        for g in group.get("groups", []):
            for stat in g.get("statisticsItems", []):
                stats[stat["name"]] = {
                    "home": stat.get("home", 0),
                    "away": stat.get("away", 0)
                }
    return stats

# ===========================
# STREAMLIT APP
# ===========================
st.set_page_config(page_title="Football AI (Worldwide)", layout="centered")

st.markdown("<h2 style='text-align: center;'>⚽ Football AI (Worldwide)</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Pick ANY league or all matches today</p>", unsafe_allow_html=True)

# League selector
league_choice = st.selectbox("🏆 Choose League", list(LEAGUES.keys()))
league = LEAGUES[league_choice]

# Get matches
matches = get_upcoming_matches(league)

if matches:
    options = [m["label"] for m in matches]
    selected = st.selectbox("📅 Choose a Match", options)

    match = next((m for m in matches if m["label"] == selected), None)

    if match:
        st.success(f"✅ {match['home']} vs {match['away']}")

        sofascore_stats = get_sofascore_stats(match["id"])

        # --- Safe defaults if no stats available ---
        home_shots = sofascore_stats.get("Shots on target", {}).get("home", 5)
        away_shots = sofascore_stats.get("Shots on target", {}).get("away", 5)
        home_poss = sofascore_stats.get("Ball possession", {}).get("home", 50)
        away_poss = sofascore_stats.get("Ball possession", {}).get("away", 50)
        home_cards = sofascore_stats.get("Yellow cards", {}).get("home", 1)
        away_cards = sofascore_stats.get("Yellow cards", {}).get("away", 1)

        # --- Dummy training data ---
        train_data = pd.DataFrame({
            "home_shots": [10, 8, 12, 7, 9],
            "away_shots": [5, 9, 7, 8, 6],
            "home_possession": [55, 62, 48, 51, 57],
            "away_possession": [45, 38, 52, 49, 43],
            "home_cards": [1, 2, 1, 3, 2],
            "away_cards": [2, 1, 2, 1, 3],
            "result": ["HomeWin", "AwayWin", "HomeWin", "Draw", "HomeWin"]
        })
        X = train_data.drop("result", axis=1)
        y = train_data["result"]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # --- Prediction ---
        new_match = pd.DataFrame({
            "home_shots": [home_shots],
            "away_shots": [away_shots],
            "home_possession": [home_poss],
            "away_possession": [away_poss],
            "home_cards": [home_cards],
            "away_cards": [away_cards]
        })

        probs = model.predict_proba(new_match)[0]
        results = dict(zip(model.classes_, probs))

        st.markdown("### 📈 Prediction Odds")
        for outcome, prob in results.items():
            color = "green" if prob > 0.5 else "orange" if prob > 0.3 else "red"
            st.markdown(
                f"<div style='padding:10px; margin:5px; border-radius:8px; background-color:{color}; color:white; text-align:center;'>"
                f"{outcome}: {prob*100:.1f}%"
                "</div>",
                unsafe_allow_html=True
            )

        st.markdown("### 📊 SofaScore Stats (if available)")
        st.json(sofascore_stats if sofascore_stats else {"message": "No stats available yet"})
else:
    st.error("❌ No matches found.")
