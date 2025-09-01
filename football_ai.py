# football_ai.py
# Copy-paste this entire file into Streamlit and run.
import streamlit as st
import requests
import math
import numpy as np
from datetime import datetime, timedelta

# -------------------------
# Configuration
# -------------------------
SOFA = "https://api.sofascore.com/api/v1"
DAYS_AHEAD = 2            # today + tomorrow
N_FORM_MATCHES = 6        # last N finished matches for form/H2H
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

st.set_page_config(page_title="Football Betting Evaluator", layout="wide")
st.title("⚽ Football Betting Evaluator")
st.caption("Predictions use recent form, H2H, home/away & travel distance. For analysis only — not financial advice.")

# -------------------------
# Built-in geolocation table (team name keys simplified)
# Add any teams you want here. If a team isn't present we fall back to neutral.
# Format: "canonical team name": (lat, lon)
# This is a small sample. Extend it with teams you care about.
# -------------------------
TEAM_GEO = {
    "Manchester United": (53.4631, -2.2913),
    "Manchester City": (53.4831, -2.2004),
    "Liverpool": (53.4308, -2.9608),
    "Arsenal": (51.5549, -0.1084),
    "Chelsea": (51.4816, -0.1910),
    "Tottenham Hotspur": (51.6043, -0.0669),
    "Real Madrid": (40.4531, -3.6883),
    "Barcelona": (41.3809, 2.1228),
    "Atletico Madrid": (40.4362, -3.5994),
    "Juventus": (45.1096, 7.6413),
    "Inter": (45.4781, 9.1231),
    "AC Milan": (45.4784, 9.2342),
    "Bayern Munich": (48.2188, 11.6247),
    "Borussia Dortmund": (51.4922, 7.4510),
    "PSG": (48.8414, 2.2530),
    "Ajax": (52.3144, 4.9413),
    "Porto": (41.1633, -8.6218),
    "Benfica": (38.7520, -9.1848),
    "RB Leipzig": (51.3470, 12.3833),
    "Monaco": (43.7347, 7.4206),
    # add more as needed...
}

# -------------------------
# Utility functions
# -------------------------
def fetch_json(url, retry=1):
    """Safe fetch with user-agent and graceful empty fallback."""
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=18)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        if retry and code in (403, 429):
            # retry once
            return fetch_json(url, retry=0)
        # return empty structure that callers can handle
        return {}
    except Exception:
        return {}

def utc_from_ts(ts):
    try:
        return datetime.utcfromtimestamp(int(ts))
    except Exception:
        return None

def haversine_km(lat1, lon1, lat2, lon2):
    # returns great-circle distance in kilometers
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def get_team_coords(team_name):
    # try direct match, else attempt simplified matching (very simple)
    if team_name in TEAM_GEO:
        return TEAM_GEO[team_name]
    # try substring match
    for key in TEAM_GEO:
        if key.lower() in team_name.lower() or team_name.lower() in key.lower():
            return TEAM_GEO[key]
    return None

def clip01(x): return max(0.0, min(1.0, float(x)))

# -------------------------
# SofaScore data functions
# -------------------------
def get_day_events(date_str):
    url = f"{SOFA}/sport/football/scheduled-events/{date_str}"
    data = fetch_json(url)
    return data.get("events", [])

def get_upcoming_events(days=DAYS_AHEAD):
    events = []
    now = datetime.utcnow()
    for i in range(days):
        d = (now.date() + timedelta(days=i)).strftime("%Y-%m-%d")
        day_events = get_day_events(d)
        for e in day_events:
            start_ts = e.get("startTimestamp")
            stt = utc_from_ts(start_ts)
            if stt and stt > now:
                events.append(e)
    return events

def team_last_events(team_id, n=N_FORM_MATCHES):
    url = f"{SOFA}/team/{team_id}/events/last/{n}"
    data = fetch_json(url)
    return data.get("events", [])

def h2h_matches(home_id, away_id, n=N_FORM_MATCHES):
    # public endpoint format used earlier; safe fallback to empty
    url = f"{SOFA}/team/{home_id}/versus/{away_id}/matches"
    data = fetch_json(url)
    return data.get("events", [])[:n]

# -------------------------
# Feature extraction
# -------------------------
def summarize_form(team_id, side_filter=None, n=N_FORM_MATCHES):
    """
    side_filter: None, 'home', or 'away' to consider only matches where team played home/away.
    Returns average goals for/against, wins/draws/losses counts.
    """
    evs = team_last_events(team_id, n)
    gf = ga = wins = draws = losses = played = 0
    for e in evs:
        home_id = e.get("homeTeam", {}).get("id")
        away_id = e.get("awayTeam", {}).get("id")
        hs = e.get("homeScore", {}).get("current", 0) or 0
        as_ = e.get("awayScore", {}).get("current", 0) or 0
        winner = e.get("winnerCode")
        if side_filter == "home" and team_id != home_id:
            continue
        if side_filter == "away" and team_id != away_id:
            continue

        played += 1
        if team_id == home_id:
            gf += hs; ga += as_
            if winner == 1: wins += 1
            elif winner == 2: losses += 1
            else: draws += 1
        elif team_id == away_id:
            gf += as_; ga += hs
            if winner == 2: wins += 1
            elif winner == 1: losses += 1
            else: draws += 1
    played = max(1, played)
    return {
        "played": played,
        "gf": gf/played,
        "ga": ga/played,
        "wins": wins,
        "draws": draws,
        "losses": losses
    }

def summarize_h2h(home_id, away_id, n=N_FORM_MATCHES):
    evs = h2h_matches(home_id, away_id, n)
    home_w = away_w = draws = 0
    for e in evs:
        winner = e.get("winnerCode")
        if winner == 1: home_w += 1
        elif winner == 2: away_w += 1
        else: draws += 1
    return {"home_w": home_w, "away_w": away_w, "draws": draws, "games": len(evs)}

# -------------------------
# Travel penalty using builtin geolocation
# -------------------------
def compute_travel_penalty(home_team_name, away_team_name):
    # returns penalty scalar to subtract from away strength (0..0.3)
    coords_h = get_team_coords(home_team_name)
    coords_a = get_team_coords(away_team_name)
    if not coords_h or not coords_a:
        return 0.0, None  # unknown -> neutral
    lat1, lon1 = coords_h; lat2, lon2 = coords_a
    dist = haversine_km(lat1, lon1, lat2, lon2)
    # penalty schedule (tweakable)
    if dist < 300:
        penalty = 0.0
    elif dist < 800:
        penalty = 0.05
    elif dist < 2000:
        penalty = 0.10
    else:
        penalty = 0.18
    return penalty, dist

# -------------------------
# Prediction engine
# -------------------------
def predict_markets(home_name, away_name, home_id, away_id, event_id=None):
    # gather form
    home_overall = summarize_form(home_id, None)
    away_overall = summarize_form(away_id, None)
    home_home = summarize_form(home_id, "home")
    away_away = summarize_form(away_id, "away")
    h2h = summarize_h2h(home_id, away_id)

    # base strength signals
    home_attack = home_home["gf"]
    home_def = home_home["ga"]
    away_attack = away_away["gf"]
    away_def = away_away["ga"]

    # wins factor
    home_wins_factor = (home_home["wins"] / max(1, home_home["played"]))
    away_wins_factor = (away_away["wins"] / max(1, away_away["played"]))

    # initial numeric strengths
    home_strength = 0.6*home_attack - 0.4*home_def + 0.5*home_wins_factor + 0.2
    away_strength = 0.6*away_attack - 0.4*away_def + 0.5*away_wins_factor - 0.05

    # H2H nudges
    if h2h["games"] > 0:
        home_strength += 0.08 * h2h["home_w"]
        away_strength += 0.08 * h2h["away_w"]

    # Travel penalty (reduces away_strength)
    travel_penalty, distance_km = compute_travel_penalty(home_name, away_name)
    away_strength -= travel_penalty

    # Draw anchor: closer strengths -> higher draw
    draw_anchor = -abs(home_strength - away_strength) * 0.25

    # Softmax to probabilities
    exps = np.exp([home_strength, draw_anchor, away_strength])
    probs = exps / exps.sum()
    p_home, p_draw, p_away = probs.tolist()

    # Goals / Over25 using simple expected goals proxy
    lam_home = max(0.1, home_attack * 0.95 + (home_overall["gf"]*0.2))
    lam_away = max(0.1, away_attack * 0.95 + (away_overall["gf"]*0.2))
    lam_total = lam_home + lam_away
    # approximate P(over 2.5) via Poisson tail (approx)
    # compute P(X >= 3) ≈ 1 - Poisson CDF(2)
    def poisson_cdf(k, lam):
        s = 0.0
        for i in range(0, k+1):
            s += math.exp(-lam) * (lam**i) / math.factorial(i)
        return s
    p_over25 = clip01(1 - poisson_cdf(2, lam_total))

    # BTTS: both teams score at least once
    # approximate P(home scores >=1) * P(away scores >=1) assuming independence
    p_home_scores = 1 - math.exp(-lam_home)
    p_away_scores = 1 - math.exp(-lam_away)
    p_btts = clip01(p_home_scores * p_away_scores)

    # Corners estimate: use team attacking tempo as proxy
    avg_corners_home = home_home["gf"] * 2.2 + home_overall["gf"]*0.3  # heuristic
    avg_corners_away = away_away["gf"] * 2.0 + away_overall["gf"]*0.3
    avg_corners_total = (avg_corners_home + avg_corners_away)
    # P(over 9.5 corners) as logistic on avg_corners_total
    p_corners_over = clip01(1 / (1 + math.exp(-(avg_corners_total - 9.5)/2.0)))

    # Cards estimate: use defensive frailty as proxy -> more fouls/cards
    avg_cards_total = (home_overall["ga"] + away_overall["ga"]) * 0.9  # heuristic scale
    p_cards_over = clip01(1 / (1 + math.exp(-(avg_cards_total - 3.5)/1.2)))  # 3.5 combined threshold

    # package outputs
    preds = {
        "1X2": {"Home": float(p_home), "Draw": float(p_draw), "Away": float(p_away)},
        "Totals_2_5": {"Over2.5": float(p_over25), "Under2.5": float(1 - p_over25)},
        "BTTS": {"Yes": float(p_btts), "No": float(1 - p_btts)},
        "Corners_9_5": {"Over9.5": float(p_corners_over), "Under9.5": float(1 - p_corners_over)},
        "Cards": {"OverEstimate": float(p_cards_over), "UnderEstimate": float(1 - p_cards_over)},
        "meta": {
            "lam_home": lam_home, "lam_away": lam_away, "lam_total": lam_total,
            "distance_km": distance_km, "travel_penalty": travel_penalty,
            "home_home_form": home_home, "away_away_form": away_away, "h2h": h2h
        }
    }
    return preds

def best_bet_from_preds(preds):
    best_mkt = None; best_opt = None; best_val = -1.0
    for mkt, opts in preds.items():
        if mkt == "meta": continue
        for opt, val in opts.items():
            if val > best_val:
                best_mkt, best_opt, best_val = mkt, opt, val
    return best_mkt, best_opt, best_val

# -------------------------
# UI / Flow
# -------------------------
st.markdown("## Upcoming matches (today & tomorrow) — filtered to future kickoff only")
events = get_upcoming_events(DAYS_AHEAD)

if not events:
    st.error("No upcoming matches found for the selected window. Try again later.")
    st.stop()

# tournament filter list
tournaments = sorted(list({ (e.get("tournament") or {}).get("name","Unknown") for e in events }))
tournaments = ["All"] + tournaments
sel_tournament = st.selectbox("Filter by tournament (All = show all)", tournaments, index=0)

filtered = events if sel_tournament == "All" else [e for e in events if ((e.get("tournament") or {}).get("name","") == sel_tournament)]
if not filtered:
    st.warning("No matches for that tournament in the upcoming window. Showing all instead.")
    filtered = events

# show select
def label_event(e):
    h = e.get("homeTeam", {}).get("name","?")
    a = e.get("awayTeam", {}).get("name","?")
    t = utc_from_ts(e.get("startTimestamp"))
    tstr = t.strftime("%Y-%m-%d %H:%M UTC") if t else "?"
    return f"{h} vs {a} • {tstr} • {((e.get('tournament') or {}).get('name',''))}"

chosen = st.selectbox("Choose match", filtered, format_func=label_event)

if chosen:
    home = chosen.get("homeTeam", {}); away = chosen.get("awayTeam", {})
    hname = home.get("name",""); aname = away.get("name","")
    hid = home.get("id"); aid = away.get("id")
    event_id = chosen.get("id")
    st.markdown(f"### Selected: {hname} vs {aname}")
    st.caption(f"Tournament: {((chosen.get('tournament') or {}).get('name',''))}")

    with st.spinner("Computing predictions..."):
        preds = predict_markets(hname, aname, hid, aid, event_id)
        best_mkt, best_opt, best_val = best_bet_from_preds(preds)

    # show meta
    meta = preds.get("meta", {})
    st.markdown("**Context (form / travel)**")
    cols = st.columns(3)
    with cols[0]:
        st.write("Home (recent home-only):")
        hh = meta.get("home_home_form", {})
        st.write(f"W-D-L: {hh.get('wins',0)}-{hh.get('draws',0)}-{hh.get('losses',0)}")
        st.write(f"GF/GA (per home game): {hh.get('gf',0):.2f} / {hh.get('ga',0):.2f}")
    with cols[1]:
        st.write("Away (recent away-only):")
        aa = meta.get("away_away_form", {})
        st.write(f"W-D-L: {aa.get('wins',0)}-{aa.get('draws',0)}-{aa.get('losses',0)}")
        st.write(f"GF/GA (per away game): {aa.get('gf',0):.2f} / {aa.get('ga',0):.2f}")
    with cols[2]:
        dist = meta.get("distance_km")
        if dist is None:
            st.write("Travel distance: unknown (no geo data)")
        else:
            st.write(f"Travel distance: {dist:.0f} km")
            st.write(f"Travel penalty (applied to away strength): {meta.get('travel_penalty',0):.2f}")

    st.markdown("---")
    st.markdown("## 📈 Probabilities (model estimates)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("1X2")
        st.write({k: f"{v*100:.1f}%" for k,v in preds["1X2"].items()})
    with c2:
        st.subheader("Totals (2.5)")
        st.write({k: f"{v*100:.1f}%" for k,v in preds["Totals_2_5"].items()})
        st.subheader("BTTS")
        st.write({k: f"{v*100:.1f}%" for k,v in preds["BTTS"].items()})
    with c3:
        st.subheader("Corners (9.5)")
        st.write({k: f"{v*100:.1f}%" for k,v in preds["Corners_9_5"].items()})
        st.subheader("Cards (estimate)")
        st.write({k: f"{v*100:.1f}%" for k,v in preds["Cards"].items()})

    st.markdown("---")
    st.subheader("🎯 Safe Bet Suggestion")
    if best_mkt:
        st.success(f"Suggested: **{best_mkt} → {best_opt}** ({best_val*100:.1f}%)")
        st.caption("This selects the single market option with the highest model probability/confidence.")
    else:
        st.info("No strong suggestion available.")

    st.markdown("---")
    st.caption("Model is heuristic and transparent (uses past form, H2H and travel). Not financial advice — bet responsibly.")
