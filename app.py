import streamlit as st
import pandas as pd
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import time

st.set_page_config(page_title="NBA Edge Predictor", layout="centered")
st.title("🏀 NBA Edge Predictor")
st.caption("Compare model-predicted player points vs sportsbook lines to find value.")

# ---------------------------
# ℹ️ How this model works
# ---------------------------
with st.expander("ℹ️ How this model works"):
    st.markdown("""
    - 📦 **Data Source:** Uses `nba_api` to pull player game logs.
    - 🔄 **Seasons:** Pulls games from 2022–23, 2023–24, and 2024–25.
    - 🔢 **User-Controlled:** You pick how many recent games (up to 30) to include.
    - 🎯 **Optional Filter:** Choose an opponent to filter predictions by matchup.
    - 🧠 **Model Type:** Trains an XGBoost regression model to predict points.
    - 🧮 **Prediction:** Compares prediction to a sportsbook line to detect value.
    """)

# ---------------------------
# Load teams and players
# ---------------------------
@st.cache_data
def load_teams():
    return sorted(teams.get_teams(), key=lambda x: x['full_name'])

@st.cache_data
def load_players():
    return players.get_active_players()

teams_list = load_teams()
players_list = load_players()

team_names = [team['full_name'] for team in teams_list]
selected_team = st.selectbox("Select a Team", team_names)

team_id = next((t['id'] for t in teams_list if t['full_name'] == selected_team), None)
team_players = [p for p in players_list if p.get('team_id') == team_id]
player_names = sorted([p['full_name'] for p in team_players])

selected_player = st.selectbox("Select a Player", player_names)

# ---------------------------
# Get player ID
# ---------------------------
def get_player_id(name):
    result = players.find_players_by_full_name(name)
    return result[0]['id'] if result else None

# ---------------------------
# Pull multiple seasons and limit by recent games
# ---------------------------
def get_player_games(player_id, max_games=30, opponent=None):
    seasons = ['2024-25', '2023-24', '2022-23']
    frames = []
    for season in seasons:
        try:
            time.sleep(0.6)
            gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
            df = gamelog.get_data_frames()[0]
            if not df.empty:
                df['Season'] = season
                frames.append(df)
        except:
            continue
    if not frames:
        return pd.DataFrame()
    
    all_games = pd.concat(frames).sort_values(by='GAME_DATE', ascending=False)
    if opponent:
        all_games = all_games[all_games['MATCHUP'].str.contains(opponent)]
    return all_games.head(max_games)

# ---------------------------
# Train model
# ---------------------------
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

# ---------------------------
# Controls for prediction
# ---------------------------
if selected_player:
    player_id = get_player_id(selected_player)

    st.subheader("📊 Game History Filters")
    max_games = st.slider("How many recent games to include?", 5, 30, 15)
    
    # Pull all opponent matchups
    opponent_df = get_player_games(player_id, max_games=100)
    opponents = sorted(opponent_df['MATCHUP'].str.extract(r'@ (.*)|vs\. (.*)')[0].dropna().unique())
    selected_opponent = st.selectbox("Optional: Filter by Opponent", ["All Opponents"] + opponents)
    opponent_filter = None if selected_opponent == "All Opponents" else selected_opponent

    sportsbook_line = st.number_input("📈 Enter Sportsbook Line (PTS)", 0.0, 60.0, 22.5)

    if st.button("Predict"):
        with st.spinner("Pulling data and training model..."):
            df = get_player_games(player_id, max_games=max_games, opponent=opponent_filter)
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
                st.warning("Not enough data to train the model. Try a different player or increase game count.")
