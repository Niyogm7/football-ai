import streamlit as st
import requests
from datetime import datetime, timedelta

# ============================
# CONFIG
# ============================
SOFA = "https://api.sofascore.com/api/v1"
DAYS_AHEAD = 2

st.set_page_config(page_title="Football Betting AI", layout="wide")
st.title("⚽ Football Betting AI — Upcoming Matches")
st.caption("Form + travel + referee tendencies. For analysis only — not financial advice.")

# ============================
# HELPERS
# ============================
@st.cache_data
def fetch_json(url):
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()

def get_upcoming_matches(days_ahead=DAYS_AHEAD):
    """Fetch fixtures today + tomorrow."""
    events = []
    today = datetime.utcnow().date()
    for i in range(days_ahead):
        d = today + timedelta(days=i)
        url = f"{SOFA}/sport/football/scheduled-events/{d}"
        try:
            data = fetch_json(url)
            events.extend(data.get("events", []))
        except Exception as e:
            st.warning(f"⚠️ Could not fetch {d}: {e}")
    return events

def normalize_tournament(ev):
    try:
        return ev["tournament"]["name"]
    except:
        return "Unknown"

def display_match(ev):
    h = ev["homeTeam"]["name"]
    a = ev["awayTeam"]["name"]
    t = datetime.utcfromtimestamp(ev["startTimestamp"]).strftime("%Y-%m-%d %H:%M UTC")
    return f"{h} vs {a} ({t})"

# ============================
# MAIN
# ============================
matches = get_upcoming_matches()

if not matches:
    st.error("No matches found from SofaScore. Try again later.")
    st.stop()

# Tournament filter
tournaments = sorted(set(normalize_tournament(ev) for ev in matches))
choice = st.selectbox("Filter by tournament", ["All Matches"] + tournaments)

if choice != "All Matches":
    matches = [m for m in matches if normalize_tournament(m) == choice]

if not matches:
    st.warning("No upcoming matches found for the selected filter.")
    st.stop()

# Pick a match
match = st.selectbox("Choose an upcoming match", matches, format_func=display_match)

st.success(f"Selected: {display_match(match)} • {normalize_tournament(match)}")

# ============================
# Example prediction placeholder
# ============================
st.subheader("🔮 AI Prediction (demo)")
st.write("This section will calculate win %, corners, fouls, etc. using past data + travel evaluation.")
st.info("Coming next: integrate historical stats + betting evaluation engine.")
