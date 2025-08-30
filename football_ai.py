import streamlit as st
import requests
import datetime
import math
import numpy as np
import pandas as pd
from functools import lru_cache

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
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

# -----------------------------
# SOFASCORE QUERIES
# -----------------------------
def list_scheduled_dates(days_ahead=2):
    dates = []
    today = datetime.date.today()
    for i in range(days_ahead):
        dates.append((today + datetime.timedelta(days=i)).strftime("%Y-%m-%d"))
    return dates

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

# -----------------------------
# FEATURE EXTRACTORS (last N games)
# -----------------------------
def summarize_team_form(team_id: int):
    """
    Returns averages over last N games:
    goals for/against, corners for/against, fouls for/against,
    win/draw/loss counts, avg days rest.
    """
    evs = team_last_events(team_id, N_FORM_MATCHES)
    gf = ga = cf = ca = ff = fa = 0
    wins = draws = losses = 0
    dates = []

    for e in evs:
        eid = e.get("id")
        if not eid:
            continue
        # outcome
        winner = e.get("winnerCode")  # 1=home, 2=away, 3=draw
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

        # per-event stats (corners / fouls)
        try:
            stats = event_stats(eid)
            # Corners
            home_c = to_int(safe_get(stats, "Corner kicks", "home", default=0), 0)
            away_c = to_int(safe_get(stats, "Corner kicks", "away", default=0), 0)
            # Fouls
            home_f = to_int(safe_get(stats, "Fouls", "home", default=0), 0)
            away_f = to_int(safe_get(stats, "Fouls", "away", default=0), 0)
            if team_id == home_id:
                cf += home_c; ca += away_c
                ff += home_f; fa += away_f
            elif team_id == away_id:
                cf += away_c; ca += home_c
                ff += away_f; fa += home_f
        except Exception:
            pass

        ts = e.get("startTimestamp")
        if ts: 
            try:
                dates.append(datetime.datetime.fromtimestamp(int(ts)))
            except Exception:
                pass

    n = max(1, len(evs))
    avg = lambda x: x / n
    days_rest = 0
    if len(dates) >= 2:
        dates_sorted = sorted(dates)
        diffs = [
            (dates_sorted[i] - dates_sorted[i-1]).days
            for i in range(1, len(dates_sorted))
        ]
        if diffs:
            days_rest = sum(diffs)/len(diffs)

    return {
        "games": len(evs),
        "gf": avg(gf), "ga": avg(ga),
        "cf": avg(cf), "ca": avg(ca),
        "ff": avg(ff), "fa": avg(fa),
        "wins": wins, "draws": draws, "losses": losses,
        "days_rest": days_rest
    }

# -----------------------------
# SIMPLE MODELS
# -----------------------------
def softmax3(h, d, a):
    exps = np.exp([h, d, a])
    s = exps.sum()
    return exps / s if s > 0 else np.array([1/3, 1/3, 1/3])

def poisson_prob_ge(k, lam):
    # P(X >= k) for Poisson(lambda)
    cdf = sum((math.exp(-lam) * (lam**i) / math.factorial(i)) for i in range(k))
    return 1 - cdf

def logistic(x):
    return 1/(1+math.exp(-x))

def compute_travel_penalty(is_home: bool, country_home: str, country_away: str, away_team: bool, days_rest: float):
    """
    Travel/Fatigue proxy:
    - Home team gets +0.15
    - Away team gets -0.15
    - Country change (away side) extra -0.10
    - Low rest (<4 days) -0.10
    """
    score = 0.0
    if is_home:
        score += 0.15
    if away_team:
        score -= 0.15
        if country_home and country_away and (country_home != country_away):
            score -= 0.10
    if days_rest and days_rest < 4:
        score -= 0.10
    return score

def market_predictions(home_form, away_form, country_h, country_a):
    # 1) 1X2
    base_home = 0.5*(home_form["wins"] - home_form["losses"]) + 0.3*(home_form["gf"] - home_form["ga"])
    base_away = 0.5*(away_form["wins"] - away_form["losses"]) + 0.3*(away_form["gf"] - away_form["ga"])

    home_context = compute_travel_penalty(True, country_h, country_a, False, home_form["days_rest"])
    away_context = compute_travel_penalty(False, country_h, country_a, True, away_form["days_rest"])

    h_score = base_home + home_context
    a_score = base_away + away_context
    d_score = -abs(h_score - a_score) * 0.2  # draw more likely when teams look even

    p_home, p_draw, p_away = softmax3(h_score, d_score, a_score)

    # 2) Total Goals (Over/Under 2.5) via Poisson
    lam_home = max(0.2, home_form["gf"]) * (1 + (home_context*0.5))
    lam_away = max(0.2, away_form["gf"]) * (1 + (away_context*0.5))
    lam_tot = max(0.3, lam_home + lam_away)
    p_over25 = poisson_prob_ge(3, lam_tot)  # goals >=3
    p_under25 = 1 - p_over25

    # 3) BTTS: use both attack vs defense
    atk_h = home_form["gf"]; def_h = home_form["ga"]
    atk_a = away_form["gf"]; def_a = away_form["ga"]
    p_h_scores = logistic( (atk_h - def_a) )  # not calibrated, but monotonic
    p_a_scores = logistic( (atk_a - def_h) )
    p_btts = p_h_scores * p_a_scores

    # 4) Corners O/U 9.5 using averages
    avg_corners = max(2.0, home_form["cf"] + away_form["cf"])  # crude
    p_corners_over = logistic((avg_corners - 9.5) / 2.0)
    p_corners_under = 1 - p_corners_over

    # 5) Fouls O/U 25.5
    avg_fouls = max(5.0, home_form["ff"] + away_form["ff"])
    p_fouls_over = logistic((avg_fouls - 25.5) / 3.0)
    p_fouls_under = 1 - p_fouls_over

    return {
        "1X2": {"Home": p_home, "Draw": p_draw, "Away": p_away},
        "Totals_2_5": {"Over2.5": p_over25, "Under2.5": p_under25},
        "BTTS": {"Yes": p_btts, "No": 1 - p_btts},
        "Corners_9_5": {"Over9.5": p_corners_over, "Under9.5": p_corners_under},
        "Fouls_25_5": {"Over25.5": p_fouls_over, "Under25.5": p_fouls_under},
        "lambda_total_goals": lam_tot,
        "avg_corners": avg_corners,
        "avg_fouls": avg_fouls,
    }

def pct(x):
    return f"{x*100:.1f}%"

# -----------------------------
# UI
# -----------------------------
st.markdown("<h2 style='text-align: center;'>⚽ Betting Evaluator (Upcoming Matches)</h2>", unsafe_allow_html=True)
st.caption("Uses SofaScore public data. Predictions are heuristic, not financial advice.")

league_choice = st.selectbox("🏆 Pick a view", list(LEAGUES.keys()))

if LEAGUES[league_choice] is None:
    # all matches worldwide
    matches = get_upcoming_matches_world(days_ahead=2)
else:
    # filter by league name inside scheduled-events
    matches = get_upcoming_matches_league(LEAGUES[league_choice], days_ahead=2)

if not matches:
    st.error("No upcoming matches found for this selection.")
    st.stop()

labels = [m["label"] for m in matches]
selected = st.selectbox("📅 Choose an upcoming match", labels)
match = next(m for m in matches if m["label"] == selected)

st.success(f"Selected: {match['home']} vs {match['away']} • {match.get('tournament','')}")

# Pull past form summaries
with st.spinner("Fetching team form & stats..."):
    home_form = summarize_team_form(match["home_id"])
    away_form = summarize_team_form(match["away_id"])

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🏠 Home (last 5)")
    st.write(f"W-D-L: **{home_form['wins']}-{home_form['draws']}-{home_form['losses']}**")
    st.write(f"Goals: **{home_form['gf']:.2f} for / {home_form['ga']:.2f} against**")
    st.write(f"Corners: **{home_form['cf']:.2f} for / {home_form['ca']:.2f} against**")
    st.write(f"Fouls: **{home_form['ff']:.2f} for / {home_form['fa']:.2f} against**")
    st.write(f"Avg days rest: **{home_form['days_rest']:.1f}**")

with col2:
    st.markdown("### 🛫 Away (last 5)")
    st.write(f"W-D-L: **{away_form['wins']}-{away_form['draws']}-{away_form['losses']}**")
    st.write(f"Goals: **{away_form['gf']:.2f} for / {away_form['ga']:.2f} against**")
    st.write(f"Corners: **{away_form['cf']:.2f} for / {away_form['ca']:.2f} against**")
    st.write(f"Fouls: **{away_form['ff']:.2f} for / {away_form['fa']:.2f} against**")
    st.write(f"Avg days rest: **{away_form['days_rest']:.1f}**")

preds = market_predictions(home_form, away_form, match.get("country_h"), match.get("country_a"))

st.markdown("## 📈 Market-style Probabilities")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**1X2**")
    st.write(f"Home: **{pct(preds['1X2']['Home'])}**")
    st.write(f"Draw: **{pct(preds['1X2']['Draw'])}**")
    st.write(f"Away: **{pct(preds['1X2']['Away'])}**")
with c2:
    st.markdown("**Totals (2.5)**")
    st.write(f"Over 2.5: **{pct(preds['Totals_2_5']['Over2.5'])}**")
    st.write(f"Under 2.5: **{pct(preds['Totals_2_5']['Under2.5'])}**")
    st.caption(f"λ (Poisson total goals): {preds['lambda_total_goals']:.2f}")
with c3:
    st.markdown("**BTTS**")
    st.write(f"Yes: **{pct(preds['BTTS']['Yes'])}**")
    st.write(f"No: **{pct(preds['BTTS']['No'])}**")

c4, c5 = st.columns(2)
with c4:
    st.markdown("**Corners (9.5)**")
    st.write(f"Over 9.5: **{pct(preds['Corners_9_5']['Over9.5'])}**")
    st.write(f"Under 9.5: **{pct(preds['Corners_9_5']['Under9.5'])}**")
    st.caption(f"Avg corners (team for-for): {preds['avg_corners']:.2f}")
with c5:
    st.markdown("**Fouls (25.5)**")
    st.write(f"Over 25.5: **{pct(preds['Fouls_25_5']['Over25.5'])}**")
    st.write(f"Under 25.5: **{pct(preds['Fouls_25_5']['Under25.5'])}**")
    st.caption(f"Avg fouls (team for-for): {preds['avg_fouls']:.2f}")

st.divider()
st.caption("This tool is for analysis/entertainment. Always bet responsibly.")
