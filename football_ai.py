import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier

# ===========================
# CONFIG
# ===========================
API_KEY = "YOUR_API_KEY_HERE"  # 👈 Replace with your API-Football key
HEADERS = {
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
    "X-RapidAPI-Key": API_KEY
}

# ===========================
# FUNCTIONS
# ===========================
def fetch_upcoming_matches():
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures?next=5&league=39&season=2024"
    resp = requests.get(url, headers=HEADERS).json()
    matches = []
    for item in resp.get("response", []):
        matches.append({
            "fixture_id": item["fixture"]["id"],
            "home": item["teams"]["home"]["name"],
            "away": item["teams"]["away"]["name"]
        })
    return matches

# ===========================
# STREAMLIT APP
# ===========================
st.title("⚽ Football Match Predictor AI")

matches = fetch_upcoming_matches()
if matches:
    selected = st.selectbox(
        "Select a Match", 
        options=[f"{m['home']} vs {m['away']}" for m in matches]
    )
    chosen = matches[[f"{m['home']} vs {m['away']}" for m in matches].index(selected)]

    st.write(f"You selected: {selected}")

    # Dummy dataset (replace with real historical stats later)
    train_data = pd.DataFrame({
        "home_goals_avg": [2.1, 1.8, 2.4, 1.9, 2.3],
        "away_goals_avg": [1.4, 2.0, 1.5, 2.2, 1.6],
        "home_cards_avg": [1.5, 2.0, 1.3, 1.2, 1.6],
        "away_cards_avg": [2.1, 1.7, 1.9, 2.3, 2.0],
        "result": ["HomeWin", "AwayWin", "HomeWin", "Draw", "HomeWin"]
    })

    X = train_data.drop("result", axis=1)
    y = train_data["result"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Example values (later you can pull real stats here)
    new_match = pd.DataFrame({
        "home_goals_avg":[2.0],
        "away_goals_avg":[1.4],
        "home_cards_avg":[1.5],
        "away_cards_avg":[2.0]
    })

    probs = model.predict_proba(new_match)[0]
    results = dict(zip(model.classes_, probs))

    st.subheader("📈 Prediction Probabilities")
    for outcome, prob in results.items():
        st.write(f"{outcome}: **{prob*100:.1f}%**")
else:
    st.error("No matches found. Check API key or free plan limits.")
