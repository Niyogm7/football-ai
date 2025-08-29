import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier

# ===========================
# CONFIG
# ===========================
API_KEY = st.secrets["API_KEY"]   # API key from Streamlit Secrets
HEADERS = {
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
    "X-RapidAPI-Key": API_KEY
}

# ===========================
# API-FOOTBALL FUNCTIONS
# ===========================
def search_apifootball_match(query):
    # Get next 50 matches (any league, any season)
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures?next=50"
    resp = requests.get(url, headers=HEADERS).json()

    for item in resp.get("response", []):
        home = item["teams"]["home"]["name"]
        away = item["teams"]["away"]["name"]

        # Flexible matching (ignore case + spaces)
        match_name = f"{home} vs {away}".lower().replace(" ", "")
        search_query = query.lower().replace(" ", "")

        if search_query in match_name:
            return {
                "home": home,
                "away": away,
                "home_id": item["teams"]["home"]["id"],
                "away_id": item["teams"]["away"]["id"],
                "date": item["fixture"]["date"]
            }
    return None

def get_apifootball_team_stats(team_id):
    url = f"https://api-football-v1.p.rapidapi.com/v3/teams/statistics?league=39&season=2024&team={team_id}"
    resp = requests.get(url, headers=HEADERS).json()
    stats = resp.get("response", {})
    return {
        "avg_goals": stats.get("goals", {}).get("for", {}).get("average", {}).get("total", 1.5),
        "avg_cards": stats.get("cards", {}).get("yellow", {}).get("average", {}).get("total", 1.5)
    }

# ===========================
# STREAMLIT APP
# ===========================
st.set_page_config(page_title="Football AI", layout="centered")

st.markdown("<h2 style='text-align: center;'>⚽ Football AI Predictor</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Type a match and get instant predictions</p>", unsafe_allow_html=True)

query = st.text_input("🔍 Enter Match (e.g. 'Man Utd vs Arsenal')", "")

if query:
    match = search_apifootball_match(query)

    if match:
        st.success(f"✅ {match['home']} vs {match['away']} on {match['date'][:10]}")
    else:
        st.error("❌ Match not found. Try another name (e.g. 'man utd' instead of 'manchester united').")
