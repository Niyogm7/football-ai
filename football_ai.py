import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier

# ===========================
# CONFIG
# ===========================
API_KEY = st.secrets["API_KEY"]
HEADERS = {
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
    "X-RapidAPI-Key": API_KEY
}

# ===========================
# API-FOOTBALL FUNCTIONS
# ===========================
def get_upcoming_matches():
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures?next=20"  # next 20 matches
    resp = requests.get(url, headers=HEADERS).json()
    matches = []
    for item in resp.get("response", []):
        home = item["teams"]["home"]["name"]
        away = item["teams"]["away"]["name"]
        date = item["fixture"]["date"][:10]
        matches.append({
            "label": f"{home} vs {away} ({date})",
            "home": home,
            "away": away,
            "home_id": item["teams"]["home"]["id"],
            "away_id": item["teams"]["away"]["id"],
            "date": item["fixture"]["date"]
        })
    return matches

def get_apifootball_team_stats(team_id):
    url = f"https://api-football-v1.p.rapidapi.com/v3/teams/statistics?league=39&season=2024&team={team_id}"
    resp = requests.get(url, headers=HEADERS).json()
    stats = resp.get("response", {})
    return {
        "avg_goals": stats.get("goals", {}).get("for", {}).get("average", {}).get("total", 1.5),
        "avg_cards": stats.get("cards", {}).get("yellow", {}).get("average", {}).get("total", 1.5)
    }

# ===========================
# SOFASCORE FUNCTIONS
# ===========================
def find_sofascore_match(home, away):
    url = "https://api.sofascore.com/api/v1/unique-tournament/17/season/41886/events"  # EPL season
    resp = requests.get(url).json()
    for m in resp.get("events", []):
        if (home.lower() in m["homeTeam"]["name"].lower() and 
            away.lower() in m["awayTeam"]["name"].lower()):
            return m["id"]
    return None

def get_sofascore_stats(match_id):
    url = f"https://api.sofascore.com/api/v1/event/{match_id}/statistics"
    resp = requests.get(url).json()
    stats = {}
    for group in resp.get("statistics", []):
        for item in group["groups"]:
            for stat in item["statisticsItems"]:
                stats[stat["name"]] = {
                    "home": stat["home"],
                    "away": stat["away"]
                }
    return stats

# ===========================
# STREAMLIT APP
# ===========================
st.set_page_config(page_title="Football AI", layout="centered")

st.markdown("<h2 style='text-align: center;'>⚽ Football AI Predictor</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Select an upcoming fixture to get predictions</p>", unsafe_allow_html=True)

# Get list of matches
matches = get_upcoming_matches()

if matches:
    options = [m["label"] for m in matches]
    selected = st.selectbox("📅 Choose a Match", options)

    match = next((m for m in matches if m["label"] == selected), None)

    if match:
        st.success(f"✅ {match['home']} vs {match['away']} on {match['date'][:10]}")

        # --- Get stats from API-Football ---
        home_stats = get_apifootball_team_stats(match["home_id"])
        away_stats = get_apifootball_team_stats(match["away_id"])

        # --- Get stats from SofaScore ---
        sofa_id = find_sofascore_match(match["home"], match["away"])
        if sofa_id:
            sofascore_stats = get_sofascore_stats(sofa_id)
        else:
            sofascore_stats = {}

        # --- Train dummy model ---
        train_data = pd.DataFrame({
            "home_goals": [2.1, 1.8, 2.4, 1.9, 2.3],
            "away_goals": [1.4, 2.0, 1.5, 2.2, 1.6],
            "home_cards": [1.5, 2.0, 1.3, 1.2, 1.6],
            "away_cards": [2.1, 1.7, 1.9, 2.3, 2.0],
            "home_shots": [5, 8, 6, 7, 9],
            "away_shots": [3, 6, 7, 8, 4],
            "home_possession": [55, 62, 48, 51, 57],
            "away_possession": [45, 38, 52, 49, 43],
            "result": ["HomeWin", "AwayWin", "HomeWin", "Draw", "HomeWin"]
        })
        X = train_data.drop("result", axis=1)
        y = train_data["result"]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # --- Merge features ---
        home_goals = float(home_stats["avg_goals"])
        away_goals = float(away_stats["avg_goals"])
        home_cards = float(home_stats["avg_cards"])
        away_cards = float(away_stats["avg_cards"])
        home_shots = sofascore_stats.get("Shots on target", {}).get("home", 5)
        away_shots = sofascore_stats.get("Shots on target", {}).get("away", 4)
        home_poss = sofascore_stats.get("Ball possession", {}).get("home", 50)
        away_poss = sofascore_stats.get("Ball possession", {}).get("away", 50)

        new_match = pd.DataFrame({
            "home_goals": [home_goals],
            "away_goals": [away_goals],
            "home_cards": [home_cards],
            "away_cards": [away_cards],
            "home_shots": [home_shots],
            "away_shots": [away_shots],
            "home_possession": [home_poss],
            "away_possession": [away_poss]
        })

        # --- Predict ---
        probs = model.predict_proba(new_match)[0]
        results = dict(zip(model.classes_, probs))

        st.markdown("### 📈 Prediction Odds (Combined Stats)")
        for outcome, prob in results.items():
            color = "green" if prob > 0.5 else "orange" if prob > 0.3 else "red"
            st.markdown(
                f"<div style='padding:10px; margin:5px; border-radius:8px; background-color:{color}; color:white; text-align:center;'>"
                f"{outcome}: {prob*100:.1f}%"
                "</div>",
                unsafe_allow_html=True
            )

        # --- Show SofaScore stats for context ---
        if sofascore_stats:
            st.markdown("### 📊 Key SofaScore Stats")
            st.write(sofascore_stats)
else:
    st.error("❌ No upcoming matches found (maybe API quota issue)")
