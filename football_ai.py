import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier

# ===========================
# CONFIG
# ===========================
API_KEY = "YOUR_API_KEY_HERE"  # 👈 Replace with API-Football key
HEADERS = {
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
    "X-RapidAPI-Key": API_KEY
}

# ===========================
# API-FOOTBALL FUNCTIONS
# ===========================
def get_apifootball_matches():
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures?next=5&league=39&season=2024"
    resp = requests.get(url, headers=HEADERS).json()
    matches = []
    for item in resp.get("response", []):
        matches.append({
            "id": item["fixture"]["id"],
            "home": item["teams"]["home"]["name"],
            "away": item["teams"]["away"]["name"],
            "home_id": item["teams"]["home"]["id"],
            "away_id": item["teams"]["away"]["id"]
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
st.title("⚽ Combined Football AI (API-Football + SofaScore)")

matches = get_apifootball_matches()

if matches:
    selected = st.selectbox("Select a Match", [f"{m['home']} vs {m['away']}" for m in matches])
    chosen = matches[[f"{m['home']} vs {m['away']}" for m in matches].index(selected)]

    st.write(f"You selected: {selected}")

    # Get team stats from API-Football
    home_stats = get_apifootball_team_stats(chosen["home_id"])
    away_stats = get_apifootball_team_stats(chosen["away_id"])

    # Get SofaScore live stats
    sofascore_stats = get_sofascore_stats(chosen["id"])
    st.write("📊 SofaScore Stats (latest):", sofascore_stats)

    # Training dataset (toy example for demo)
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

    # Merge API-Football + SofaScore for new match
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

    probs = model.predict_proba(new_match)[0]
    results = dict(zip(model.classes_, probs))

    st.subheader("📈 Prediction Probabilities (Combined Data)")
    for outcome, prob in results.items():
        st.write(f"{outcome}: **{prob*100:.1f}%**")
else:
    st.error("No matches found.")
