import asyncio
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sofascore_wrapper.api import SofascoreAPI
from sofascore_wrapper.search import Search

async def fetch_match_data():
    api = SofascoreAPI()
    search = Search(api, search_string="Manchester United vs Liverpool")
    match = await search.search_all()
    await api.close()
    return match

def prepare_data(match):
    data = {
        'home_team': match['homeTeam']['name'],
        'away_team': match['awayTeam']['name'],
        'home_goals': match['homeScore']['current'],
        'away_goals': match['awayScore']['current'],
        'home_possession': match['homeStats']['possession'],
        'away_possession': match['awayStats']['possession'],
        'home_shots': match['homeStats']['shots'],
        'away_shots': match['awayStats']['shots'],
    }
    return pd.DataFrame([data])

def train_model():
    # Sample historical data
    data = {
        'home_possession': [55, 60, 50, 65, 58],
        'away_possession': [45, 40, 50, 35, 42],
        'home_shots': [10, 12, 8, 14, 11],
        'away_shots': [5, 4, 6, 3, 5],
        'outcome': [1, 1, 0, 1, 0]  # 1 = Home Win, 0 = Away Win
    }
    df = pd.DataFrame(data)
    X = df[['home_possession', 'away_possession', 'home_shots', 'away_shots']]
    y = df['outcome']
    model = LogisticRegression()
    model.fit(X, y)
    return model

def predict_outcome(model, match_data):
    X_new = match_data[['home_possession', 'away_possession', 'home_shots', 'away_shots']]
    prediction = model.predict(X_new)
    return prediction[0]

async def main():
    match = await fetch_match_data()
    match_data = prepare_data(match)
    model = train_model()
    prediction = predict_outcome(model, match_data)
    outcome = "Home Win" if prediction == 1 else "Away Win"
    print(f"Predicted Outcome: {outcome}")

if __name__ == "__main__":
    asyncio.run(main())
import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Function to fetch odds data
def fetch_odds_data():
    url = "https://api.the-odds-api.com/v4/sports/soccer/odds"
    params = {
        'apiKey': 'YOUR_API_KEY',
        'regions': 'us',  # Specify the region
        'markets': 'h2h',  # Head-to-head odds
        'oddsFormat': 'decimal'
    }
    response = requests.get(url, params=params)
    return response.json()

# Function to calculate travel distance (example: using Haversine formula)
def calculate_travel_distance(home_coords, away_coords):
    # Implement Haversine formula or use a geopy library to calculate distance
    pass

# Function to prepare data
def prepare_data(match, odds_data, travel_distance):
    # Extract relevant features from match data and odds_data
    data = {
        'home_possession': match['home_possession'],
        'away_possession': match['away_possession'],
        'home_shots': match['home_shots'],
        'away_shots': match['away_shots'],
        'home_odds': odds_data['home'],
        'away_odds': odds_data['away'],
        'travel_distance': travel_distance
    }
    return pd.DataFrame([data])

# Function to train model
def train_model():
    data = {
        'home_possession': [55, 60, 50, 65, 58],
        'away_possession': [45, 40, 50, 35, 42],
        'home_shots': [10, 12, 8, 14, 11],
        'away_shots': [5, 4, 6, 3, 5],
        'home_odds': [1.5, 2.0, 1.8, 1.6, 2.1],
        'away_odds': [2.5, 3.0, 2.8, 2.6, 3.1],
        'travel_distance': [100, 200, 150, 300, 250],
        'outcome': [1, 1, 0, 1, 0]  # 1 = Home Win, 0 = Away Win
    }
    df = pd.DataFrame(data)
    X = df[['home_possession', 'away_possession', 'home_shots', 'away_shots', 'home_odds', 'away_odds', 'travel_distance']]
    y = df['outcome']
    model = LogisticRegression()
    model.fit(X, y)
    return model

# Function to predict outcome
def predict_outcome(model, match_data, odds_data, travel_distance):
    X_new = match_data[['home_possession', 'away_possession', 'home_shots', 'away_shots', 'home_odds', 'away_odds', 'travel_distance']]
    prediction = model.predict(X_new)
    return prediction[0]

# Main function
def main():
    # Fetch match data (replace with actual data fetching logic)
    match = {
        'home_possession': 55,
        'away_possession': 45,
        'home_shots': 10,
        'away_shots': 5
    }

    # Fetch odds data
    odds_data = fetch_odds_data()

    # Calculate travel distance (replace with actual coordinates)
    home_coords = (51.5074, -0.1278)  # Example: London
    away_coords = (48.8566, 2.3522)  # Example: Paris
    travel_distance = calculate_travel_distance(home_coords, away_coords)

    # Prepare data
    match_data = prepare_data(match, odds_data, travel_distance)

    # Train model
    model = train_model()

    # Predict outcome
    prediction = predict_outcome(model, match_data, odds_data, travel_distance)
    outcome = "Home Win" if prediction == 1 else "Away Win"
    print(f"Predicted Outcome: {outcome}")

if __name__ == "__main__":
    main()
