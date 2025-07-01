import streamlit as st
import pandas as pd
import time
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog
from sklearn.metrics import mean_squared_error
import xgboost as xgb

st.set_page_config(page_title="NBA Edge Predictor", layout="centered")
st.title("🏀 NBA Edge Predictor")

with st.expander("ℹ️ How this model works"):
    st.markdown("""
    - 🧠 Model: Trains an XGBoost regression model using player stats
    - 📅 Season: Only pulls data from **2024–25**
    - 🎯 Prediction: Based on average stats from recent games
    - 🏀 Opponent Filter: Optional — shows matchup-specific edge
    - ⚖️ Compares to sportsbook line to find potential value
    """)

# -------------------------------
# Load Teams and Players
# -------------------------------
@st.cache_data
def get_teams():
    return sorted(teams.get_teams(), key=lambda x: x['full_name'])

@st.cache_data
def get_active_players():
    return players.get_active_players()

teams_list = get_teams()
players_list = get_active_players()

team_names = [t['full_name'] for t in teams_list]
selected_team = st.selectbox("Select a Team", team_names)

team_abbr = next((t['abbreviation'] for t in teams_list if t['full_name'] == selected_team), None)

# -------------------------------
# Filter Players Based on 2024–25 Games with Team
# -------------------------------
@st.cache_data
def get_team_players(team_abbr):
    matched_players = []
    for p in players_list:
        try:
            time.sleep(0.2)
            gamelog = playergamelog.PlayerGameLog(player_id=p['id'], season='2024-25')
            df = gamelog.get_data_frames()[0]
            if not df.empty and any(team_abbr in matchup for matchup in df['MATCHUP'].values):
                matched_players.append({'name': p['full_name'], 'id': p['id']})
        except:
            continue
    return matched_players

with st.spinner("Loading active players..."):
    active_team_players = get_team_players(team_abbr)

if not active_team_players:
    st.warning("No players found for this team yet in 2024–25.")
    st.stop()

player_names = [p['name'] for p in active_team_players]
selected_player = st.selectbox("Select a Player", player_names)

# -------------------------------
# Get Player Game Log (single season)
# -------------------------------
def get_player_games(player_id, max_games=30, opponent=None):
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season='2024-25')
        df = gamelog.get_data_frames()[0]
        df = df.sort_values('GAME_DATE', ascending=False)
        if opponent:
            df = df[df['MATCHUP'].str.contains(opponent)]
        return df.head(max_games)
    except:
        return pd.DataFrame()

def train_model(df):
    df = df.rename(columns={'MATCHUP': 'Opponent', 'MIN': 'Minutes'})
    df['Home'] = df['Opponent'].apply(lambda x: 1 if 'vs.' in x else 0)
    df['DaysRest'] = (pd.to_datetime(df['GAME_DATE']) - pd.to_datetime(df['GAME_DATE']).shift(-1)).dt.days.fillna(2)
    df = df[['Minutes', 'FGA', 'FG3A', 'AST', 'REB', 'TOV', 'Home', 'DaysRest', 'PTS']].dropna()
    if len(df) < 5:
        return None, None
    X = df.drop(columns='PTS')
    y = df['PTS']
    model = xgb.XGBRegressor()
    model.fit(X, y)
    avg_input = pd.DataFrame([X.mean()])
    return model, avg_input

# -------------------------------
# Game & Opponent Filter UI
# -------------------------------
selected_player_id = next((p['id'] for p in active_team_players if p['name'] == selected_player), None)

st.subheader("📊 Game Filters")
max_games = st.slider("Number of recent games to include", 5, 30, 15)

df_full = get_player_games(selected_player_id, max_games=30)
opponents = sorted(set(df_full['MATCHUP'].str.extract(r'@ (.*)|vs\. (.*)')[0].dropna()))
selected_opponent = st.selectbox("Optional: Filter by Opponent", ["All Opponents"] + opponents)
opp_filter = None if selected_opponent == "All Opponents" else selected_opponent

sportsbook_line = st.number_input("📈 Sportsbook Line (PTS)", 0.0, 60.0, 22.5)

# -------------------------------
# Predict
# -------------------------------
if st.button("Predict"):
    with st.spinner("Training model and predicting..."):
        df = get_player_games(selected_player_id, max_games=max_games, opponent=opp_filter)
        model, avg_input = train_model(df)
        if model:
            prediction = model.predict(avg_input)[0]
            edge = prediction - sportsbook_line

            st.markdown(f"### 🔮 Predicted Points: `{prediction:.2f}`")
            st.markdown(f"### 📊 Sportsbook Line: `{sportsbook_line}`")
            st.markdown(f"### 🧮 Edge: `{edge:.2f}`")

            if edge > 1:
                st.success("🔥 VALUE BET: Take the OVER")
            elif edge < -1:
                st.error("🧊 VALUE BET: Take the UNDER")
            else:
                st.info("❌ No clear edge.")
        else:
            st.warning("Not enough data to train model.")
