import streamlit as st
import requests
import datetime

# ===============================
# CONFIG
# ===============================
BASE_URL = "https://api.sofascore.com/api/v1"

# ===============================
# HELPER FUNCTIONS
# ===============================
@st.cache_data
def fetch_json(url):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()

def get_upcoming_matches(days_ahead=2):
    """Fetch only upcoming football matches for today/tomorrow."""
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

def get_match_stats(event_id):
    """Fetch match statistics if available."""
    url = f"{BASE_URL}/event/{event_id}/statistics"
    try:
        return fetch_json(url)
    except:
        return {}

# ===============================
# STREAMLIT APP
# ===============================
st.title("⚽ Football Betting AI — Upcoming Matches")
st.caption("Form + travel + referee tendencies (experimental). Not financial advice.")

# Pick view
view = st.selectbox("📅 Pick a view", ["All Matches Today/Tomorrow 🌍"])

# Fetch matches
matches = get_upcoming_matches(days_ahead=2)

if not matches:
    st.warning("⚠️ No upcoming matches found for today/tomorrow. Try again later.")
else:
    # Dropdown to choose match
    match_options = [
        f"{m['homeTeam']['name']} vs {m['awayTeam']['name']} "
        f"({datetime.datetime.utcfromtimestamp(m['startTimestamp']).strftime('%Y-%m-%d %H:%M UTC')})"
        for m in matches
    ]
    selected = st.selectbox("🎯 Choose an upcoming match", match_options)

    if selected:
        idx = match_options.index(selected)
        match = matches[idx]
        event_id = match["id"]

        st.success(
            f"Selected: {match['homeTeam']['name']} vs {match['awayTeam']['name']} • {match['tournament']['name']}"
        )

        # Fetch team stats
        st.write("📊 Fetching team form & statistics...")
        stats = get_match_stats(event_id)

        if not stats:
            st.warning("No statistics available yet for this match.")
        else:
            st.json(stats)

        # Dummy prediction logic (placeholder until ML model is added)
        st.subheader("🔮 AI Prediction (Demo)")
        st.write("Home win chance: 45%")
        st.write("Draw chance: 25%")
        st.write("Away win chance: 30%")
