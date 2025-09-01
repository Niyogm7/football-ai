import streamlit as st
import requests
import datetime
import math
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="⚽ Betting Evaluator (SofaScore)", layout="centered")
N_FORM_MATCHES = 5  # how many previous matches to evaluate per team

LEAGUES = {
    "All Matches Today/Tomorrow 🌍": None,
    "Premier League": "Premier League",
    "La Liga": "LaLiga",
    "Serie A": "Serie A",
    "Bundesliga": "Bundesliga",
    "Champions League": "Champions League"
}

# -----------------------------
# HELPERS
# -----------------------------
def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def to_int(x, default=0):
    try:
        if isinstance(x, str):
            x = x.replace("%", "").strip()
        return int(float(x))
    except Exception:
        return default

@st.cache_data(ttl=1800)
def fetch_json(url: str):
    import time
    try:
        # add browser-like headers to avoid blocking
        r = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0"
            }
        )
        r.raise_for_status()
        return r.json()

    except requests.exceptions.HTTPError as e:
        if r.status_code in [403, 429]:
            st.warning(f"⚠️ SofaScore blocked or rate-limited ({r.status_code}) → Retrying once...")
            time.sleep(2)  # wait a bit and try again
            try:
                r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                return r.json()
            except Exception:
                return {"events": []}
        else:
            st.warning(f"⚠️ HTTP error {r.status_code} for {url}")
            return {"events": []}

    except Exception as e:
        st.warning(f"⚠️ Failed to fetch {url}: {e}")
        return {"events": []}

def list_scheduled_dates(days_ahead=2):
    dates = []
    today = datetime.date.today()
    for i in range(days_ahead):
        dates.append((today + datetime.timedelta(days=i)).strftime("%Y-%m-%d"))
    return dates

# -----------------------------
# UPCOMING MATCHES
# -----------------------------
def get_upcoming_matches_world(days_ahead=2):
    matches = []
    for date in list_scheduled_dates(days_ahead):
        url = f"https://api.sofascore.com/api/v1/sport/football/scheduled-events/{date}"
        data = fetch_json(url)
        for e in data.get("events", []):
            home = safe_get(e, "homeTeam", "name", default="Home")
            away = safe_get(e, "awayTeam", "name", default="Away")
            tid_h = safe_get(e, "homeTeam", "id")
            tid_a = safe_get(e, "awayTeam", "id")
            if tid_h is None or tid_a is None:
                continue
            matches.append({
                "label": f"{home} vs {away} ({safe_get(e,'tournament','name','',default='')}, {date})",
                "home": home, "away": away,
                "home_id": tid_h, "away_id": tid_a,
                "event_id": e.get("id"),
                "date": date,
                "country_h": safe_get(e, "homeTeam", "country", "name", default=None),
                "country_a": safe_get(e, "awayTeam", "country", "name", default=None),
                "tournament": safe_get(e, "tournament", "name", default="")
            })
    return matches

def get_upcoming_matches_league(league_name, days_ahead=2):
    matches = []
    for date in list_scheduled_dates(days_ahead):
        url = f"https://api.sofascore.com/api/v1/sport/football/scheduled-events/{date}"
        data = fetch_json(url)
        for e in data.get("events", []):
            if league_name.lower() not in safe_get(e, "tournament", "name", default="").lower():
                continue
            home = safe_get(e, "homeTeam", "name", default="Home")
            away = safe_get(e, "awayTeam", "name", default="Away")
            matches.append({
                "label": f"{home} vs {away} ({league_name}, {date})",
                "home": home,
                "away": away,
                "home_id": safe_get(e, "homeTeam", "id"),
                "away_id": safe_get(e, "awayTeam", "id"),
                "event_id": e.get("id"),
                "date": date,
                "country_h": safe_get(e, "homeTeam", "country", "name", default=None),
                "country_a": safe_get(e, "awayTeam", "country", "name", default=None),
                "tournament": safe_get(e, "tournament", "name", default="")
            })
    return matches

# -----------------------------
# TEAM STATS
# -----------------------------
@st.cache_data(ttl=1800)
def team_last_events(team_id: int, limit=N_FORM_MATCHES):
    url = f"https://api.sofascore.com/api/v1/team/{team_id}/events/last/{limit}"
    return fetch_json(url).get("events", [])

@st.cache_data(ttl=1800)
def event_stats(event_id: int):
    url = f"https://api.sofascore.com/api/v1/event/{event_id}/statistics"
    data = fetch_json(url)
    out = {}
    for group in data.get("statistics", []):
        for g in group.get("groups", []):
            for it in g.get("statisticsItems", []):
                name = it.get("name")
                if not name:
                    continue
                out[name] = {
                    "home": it.get("home", 0),
                    "away": it.get("away", 0)
                }
    return out

def summarize_team_form(team_id: int):
    evs = team_last_events(team_id, N_FORM_MATCHES)
    gf = ga = cf = ca = ff = fa = 0
    wins = draws = losses = 0

    for e in evs:
        eid = e.get("id")
        if not eid:
            continue

        # Match result
        winner = e.get("winnerCode")
        home_id = safe_get(e, "homeTeam", "id")
        away_id = safe_get(e, "awayTeam", "id")
        h_score = safe_get(e, "homeScore", "current", default=0)
        a_score = safe_get(e, "awayScore", "current", default=0)

        if team_id == home_id:
            gf += to_int(h_score, 0); ga += to_int(a_score, 0)
            if winner == 1: wins += 1
            elif winner == 3: draws += 1
            elif winner == 2: losses += 1
        elif team_id == away_id:
            gf += to_int(a_score, 0); ga += to_int(h_score, 0)
            if winner == 2: wins += 1
            elif winner == 3: draws += 1
            elif winner == 1: losses += 1

        # ✅ Only fetch stats if they exist (past matches)
        try:
            stats = event_stats(eid)
            if stats:
                home_c = to_int(safe_get(stats, "Corner kicks", "home", default=0), 0)
                away_c = to_int(safe_get(stats, "Corner kicks", "away", default=0), 0)
                home_f = to_int(safe_get(stats, "Fouls", "home", default=0), 0)
                away_f = to_int(safe_get(stats, "Fouls", "away", default=0), 0)
                if team_id == home_id:
                    cf += home_c; ca += away_c
                    ff += home_f; fa += away_f
                elif team_id == away_id:
                    cf += away_c; ca += home_c
                    ff += away_f; fa += home_f
        except Exception:
            # Ignore if no stats (upcoming matches don’t have them)
            pass

    n = max(1, len(evs))
    avg = lambda x: x / n
    return {
        "games": len(evs),
        "gf": avg(gf), "ga": avg(ga),
        "cf": avg(cf), "ca": avg(ca),
        "ff": avg(ff), "fa": avg(fa),
        "wins": wins, "draws": draws, "losses": losses,
        "days_rest": 4
    }

# -----------------------------
# PREDICTIONS
# -----------------------------
def softmax3(h, d, a):
    exps = np.exp([h, d, a])
    s = exps.sum()
    return exps / s if s > 0 else np.array([1/3, 1/3, 1/3])

def logistic(x):
    return 1/(1+math.exp(-x))

def market_predictions(home_form, away_form):
    h_score = 0.5*(home_form["wins"]-home_form["losses"]) + 0.3*(home_form["gf"]-home_form["ga"]) + 0.2
    a_score = 0.5*(away_form["wins"]-away_form["losses"]) + 0.3*(away_form["gf"]-away_form["ga"]) - 0.2
    d_score = -abs(h_score - a_score)*0.2
    p_home, p_draw, p_away = softmax3(h_score, d_score, a_score)
    p_over25 = logistic((home_form["gf"]+away_form["gf"] - 2.5))
    p_under25 = 1 - p_over25
    p_btts = logistic((home_form["gf"]-away_form["ga"]) + (away_form["gf"]-home_form["ga"]))
    avg_corners = home_form["cf"] + away_form["cf"]
    p_corners_over = logistic((avg_corners-9.5)/2)
    p_corners_under = 1-p_corners_over
    avg_fouls = home_form["ff"]+away_form["ff"]
    p_fouls_over = logistic((avg_fouls-25.5)/3)
    p_fouls_under = 1-p_fouls_over
    return {
        "1X2": {"Home": p_home, "Draw": p_draw, "Away": p_away},
        "Totals_2_5": {"Over2.5": p_over25, "Under2.5": p_under25},
        "BTTS": {"Yes": p_btts, "No": 1-p_btts},
        "Corners_9_5": {"Over9.5": p_corners_over, "Under9.5": p_corners_under},
        "Fouls_25_5": {"Over25.5": p_fouls_over, "Under25.5": p_fouls_under}
    }

def pct(x): return f"{x*100:.1f}%"

def find_best_bet(preds):
    best_market, best_option, best_prob = None, None, 0
    for market, opts in preds.items():
        for opt, prob in opts.items():
            if prob > best_prob:
                best_market, best_option, best_prob = market, opt, prob
    return best_market, best_option, best_prob

# -----------------------------
# UI
# -----------------------------
st.markdown("<h2 style='text-align: center;'>⚽ Betting Evaluator (Upcoming Matches)</h2>", unsafe_allow_html=True)

league_choice = st.selectbox("🏆 Pick a view", list(LEAGUES.keys()))
if LEAGUES[league_choice] is None:
    matches = get_upcoming_matches_world(days_ahead=2)
else:
    matches = get_upcoming_matches_league(LEAGUES[league_choice], days_ahead=2)

if not matches:
    st.error("No upcoming matches found.")
    st.stop()

labels = [m["label"] for m in matches]
selected = st.selectbox("📅 Choose an upcoming match", labels)
match = next(m for m in matches if m["label"] == selected)

st.success(f"Selected: {match['home']} vs {match['away']} • {match.get('tournament','')}")

with st.spinner("Fetching team form..."):
    home_form = summarize_team_form(match["home_id"])
    away_form = summarize_team_form(match["away_id"])

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🏠 Home (last 5)")
    st.write(f"W-D-L: {home_form['wins']}-{home_form['draws']}-{home_form['losses']}")
    st.write(f"Goals: {home_form['gf']:.2f} for / {home_form['ga']:.2f} against")
    st.write(f"Corners: {home_form['cf']:.2f} for / {home_form['ca']:.2f} against")
    st.write(f"Fouls: {home_form['ff']:.2f} for / {home_form['fa']:.2f} against")

with col2:
    st.markdown("### 🛫 Away (last 5)")
    st.write(f"W-D-L: {away_form['wins']}-{away_form['draws']}-{away_form['losses']}")
    st.write(f"Goals: {away_form['gf']:.2f} for / {away_form['ga']:.2f} against")
    st.write(f"Corners: {away_form['cf']:.2f} for / {away_form['ca']:.2f} against")
    st.write(f"Fouls: {away_form['ff']:.2f} for / {away_form['fa']:.2f} against")

preds = market_predictions(home_form, away_form)

st.markdown("## 📈 Market Probabilities")
st.write("**1X2**")
st.write({k:pct(v) for k,v in preds['1X2'].items()})
st.write("**Totals (2.5)**")
st.write({k:pct(v) for k,v in preds['Totals_2_5'].items()})
st.write("**BTTS**")
st.write({k:pct(v) for k,v in preds['BTTS'].items()})
st.write("**Corners (9.5)**")
st.write({k:pct(v) for k,v in preds['Corners_9_5'].items()})
st.write("**Fouls (25.5)**")
st.write({k:pct(v) for k,v in preds['Fouls_25_5'].items()})

# -----------------------------
# BEST BET SUGGESTION
# -----------------------------
best_market, best_option, best_prob = find_best_bet(preds)
st.markdown("## 🎯 Best Bet Suggestion")
st.success(f"Most confident: **{best_market} → {best_option}** ({best_prob*100:.1f}%)")

st.caption("For entertainment/analysis. Not financial advice.")
