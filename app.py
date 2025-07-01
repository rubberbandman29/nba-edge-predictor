import streamlit as st
import pandas as pd
import time
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog
import xgboost as xgb

st.set_page_config(page_title="NBA Edge Predictor", layout="centered")
st.title("🏀 NBA Edge Predictor")

with st.expander("ℹ️ How this model works"):
    st.markdown("""
    - ✅ Trains an XGBoost model on **2024–25** regular season stats only
    - 🔢 Uses features like minutes, FGA, AST, REB, TOV, rest days
    - 🎯 Optional: Filter prediction by opponent matchup
    - ⚖️ Compares prediction to sportsbook line to show value
    """)

# --------------------------------------
# Load data
# --------------------------------------
@st.cache_data
def get_all_players():
    return players.get_active_players()

@st.cache_data
def get_all_teams():
    return sorted(teams.get_teams(), key=lambda x: x['full_name'])

player_list = get_all_players()
team_list = get_all_teams()
team_names = [t['full_name'] for t in team_list]

selected_team = st.selectbox("Select a Team", team_names)

# --------------------------------------
# Filter players based on 2024–25 games
# --------------------------------------
@st.cache_data
def get_team_players_by_gamelog(team_name):
    matched = []
    for p in player_list:
        try:
            time.sleep(0.3)  # To avoid rate limits
            gamelog = playergamelog.PlayerGameLog(player_id=p['id'], season='2024-25')
            df = gamelog.get_data_frames()[0]
            if not df.empty and any(team_name.split()[0] in m for m in df['MATCHUP']):
                matched.append({'name': p['full_name'], 'id': p['id']})
        except:
            continue
    return matched

with st.spinner("Loading players..."):
    players_found = get_team_players_by_gamelog(selected_team)

if not players_found:
    st.warning("No player game data found yet for this team in 2024–25.")
    st.stop()

player_names = sorted([p['name'] for p in players_found])
selected_player = st.selectbox("Select a Player", player_names)
selected_player_id = next(p['id'] for p in players_found if p['name'] == selected_player)

# --------------------------------------
# Game log + model
# --------------------------------------
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

# --------------------------------------
# UI filters
# --------------------------------------
st.subheader("📊 Game Filters")
max_games = st.slider("Number of recent games to include", 5, 30, 15)

df_full = get_player_games(selected_player_id, max_games=30)
opponents = sorted(set(df_full['MATCHUP'].str.extract(r'@ (.*)|vs\. (.*)')[0].dropna()))
selected_opponent = st.selectbox("Optional: Filter by Opponent", ["All Opponents"] + opponents)
opp_filter = None if selected_opponent == "All Opponents" else selected_opponent

sportsbook_line = st.number_input("📈 Sportsbook Line (PTS)", 0.0, 60.0, 22.5)

# --------------------------------------
# Prediction
# --------------------------------------
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
