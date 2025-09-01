import streamlit as st
import requests
import datetime
import math

BASE_URL = "https://api.sofascore.com/api/v1"

st.set_page_config(page_title="Football Betting AI", layout="wide")
st.title("⚽ Football Betting AI — Upcoming Matches")
st.caption("Based on recent form, head-to-head, and home/away factors. Not financial advice.")

# ============================================================
# HELPERS
# ============================================================
@st.cache_data
def fetch_json(url):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        return r.json()
    except:
        return {}

def get_upcoming_matches(days_ahead=2):
    matches = []
    now = datetime.datetime.utcnow()
    for i in range(days_ahead):
        date = (now + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        url = f"{BASE_URL}/sport/football/scheduled-events/{date}"
        data = fetch_json(url)
        for ev in data.get("events", []):
            start_time = datetime.datetime.utcfromtimestamp(ev["startTimestamp"])
            if start_time > now:  # ✅ only future matches
                matches.append(ev)
    return matches

def get_team_form(team_id, n=5):
    """Fetch last n matches for a team"""
    url = f"{BASE_URL}/team/{team_id}/events/last/{n}"
    data = fetch_json(url)
    return data.get("events", [])

def get_h2h(home_id, away_id, n=5):
    """Fetch head-to-head history"""
    url = f"{BASE_URL}/team/{home_id}/versus/{away_id}/matches"
    data = fetch_json(url)
    return data.get("events", [])[:n]

# ============================================================
# ANALYSIS
# ============================================================
def analyze_team_form(team_id, n=5, home=False):
    matches = get_team_form(team_id, n)
    gf = ga = wins = draws = losses = 0
    home_games = away_games = 0
    for m in matches:
        h_id = m["homeTeam"]["id"]
        a_id = m["awayTeam"]["id"]
        h_score = m.get("homeScore", {}).get("current", 0)
        a_score = m.get("awayScore", {}).get("current", 0)
        winner = m.get("winnerCode")

        if team_id == h_id:
            gf += h_score; ga += a_score; home_games += 1
            if winner == 1: wins += 1
            elif winner == 2: losses += 1
            else: draws += 1
        elif team_id == a_id:
            gf += a_score; ga += h_score; away_games += 1
            if winner == 2: wins += 1
            elif winner == 1: losses += 1
            else: draws += 1

    games = max(1, len(matches))
    return {
        "gf": gf / games,
        "ga": ga / games,
        "wins": wins, "draws": draws, "losses": losses,
        "home_games": home_games, "away_games": away_games
    }

def analyze_h2h(home_id, away_id, n=5):
    matches = get_h2h(home_id, away_id, n)
    home_wins = away_wins = draws = 0
    for m in matches:
        winner = m.get("winnerCode")
        if winner == 1: home_wins += 1
        elif winner == 2: away_wins += 1
        else: draws += 1
    return {"home_wins": home_wins, "away_wins": away_wins, "draws": draws, "games": len(matches)}

def predict(home_id, away_id):
    home_form = analyze_team_form(home_id, n=5)
    away_form = analyze_team_form(away_id, n=5)
    h2h = analyze_h2h(home_id, away_id, n=5)

    # Base strength = attack - defense + win %
    home_strength = home_form["gf"] - home_form["ga"] + (home_form["wins"] / 5)
    away_strength = away_form["gf"] - away_form["ga"] + (away_form["wins"] / 5)

    # Head-to-head adjustment
    if h2h["games"] > 0:
        home_strength += h2h["home_wins"] * 0.2
        away_strength += h2h["away_wins"] * 0.2

    # Home advantage boost
    home_strength *= 1.15

    # Normalize to probabilities
    total = abs(home_strength) + abs(away_strength) + 1e-6
    p_home = home_strength / total
    p_away = away_strength / total
    p_draw = max(0.1, 1 - (p_home + p_away))

    # Normalize again
    s = p_home + p_draw + p_away
    p_home, p_draw, p_away = p_home/s, p_draw/s, p_away/s

    # Safest bet
    best = max([
        ("Home Win", p_home),
        ("Draw", p_draw),
        ("Away Win", p_away)
    ], key=lambda x: x[1])

    return {
        "home_win": round(p_home*100, 1),
        "draw": round(p_draw*100, 1),
        "away_win": round(p_away*100, 1),
        "best": best
    }

# ============================================================
# APP
# ============================================================
matches = get_upcoming_matches(days_ahead=2)

if not matches:
    st.error("⚠️ No upcoming matches found.")
else:
    match = st.selectbox(
        "Select a match",
        matches,
        format_func=lambda m: f"{m['homeTeam']['name']} vs {m['awayTeam']['name']} • {m['tournament']['name']}"
    )

    if match:
        home = match["homeTeam"]
        away = match["awayTeam"]
        st.success(f"Selected: {home['name']} vs {away['name']}")

        st.write("🔎 Analyzing recent form & H2H...")
        preds = predict(home["id"], away["id"])

        st.subheader("📊 Prediction Results")
        st.write(f"🏠 Home Win: **{preds['home_win']}%**")
        st.write(f"🤝 Draw: **{preds['draw']}%**")
        st.write(f"🚶 Away Win: **{preds['away_win']}%**")

        st.subheader("💡 Safest Betting Option")
        st.success(f"👉 **{preds['best'][0]}** (Confidence: {preds['best'][1]*100:.1f}%)")
