import streamlit as st
import pandas as pd
import time
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, commonteamroster
import xgboost as xgb

st.set_page_config(page_title="NBA Edge Predictor", layout="centered")
st.title("🏀 NBA Edge Predictor")

with st.expander("ℹ️ How this model works"):
    st.markdown("""
    - 🧠 XGBoost model trained on 2024–25 game logs
    - 🧮 Uses stats like minutes, FGA, AST, REB, TOV, rest days
    - 🏀 Select a team → choose player from **actual NBA roster**
    - 🔍 Optional filter by opponent
    - ⚖️ Compares prediction to sportsbook line to find value
    """)

# -------------------------------
# ✅ Load teams and rosters
# -------------------------------
@st.cache_data
def get_team_data():
    return sorted(teams.get_teams(), key=lambda x: x['full_name'])

@st.cache_data
def get_player_data():
    return players.get_players()

team_data = get_team_data()
player_data = get_player_data()

team_names = [team['full_name'] for team in team_data]
selected_team = st.selectbox("Select Team", team_names)

team_id = next(team['id'] for team in team_data if team['full_name'] == selected_team)
roster = commonteamroster.CommonTeamRoster(team_id=team_id)
roster_df = roster.get_data_frames()[0]
roster_names = roster_df['PLAYER'].tolist()

# Match full player info
active_players = [p for p in player_data if p['is_active']]
team_players = [p for p in active_players if p['full_name'] in roster_names]
player_names = sorted([p['full_name'] for p in team_players])

selected_player = st.selectbox("Select Player", player_names)
selected_player_id = next(p['id'] for p in team_players if p['full_name'] == selected_player)

# -------------------------------
# 🔁 Get 2024–25 Game Log
# -------------------------------
def get_player_games(player_id, max_games=30, opponent=None):
    try:
        time.sleep(0.5)
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
    model = xgboost.XGBRegressor()
    model.fit(X, y)
    avg_input = pd.DataFrame([X.mean()])
    return model, avg_input

# -------------------------------
# 🎛️ Filters
# -------------------------------
st.subheader("📊 Game Filters")
max_games = st.slider("Number of recent games to include", 5, 30, 15)

df_full = get_player_games(selected_player_id, max_games=30)
opponents = sorted(set(df_full['MATCHUP'].str.extract(r'@ (.*)|vs\. (.*)')[0].dropna()))
selected_opponent = st.selectbox("Optional: Filter by Opponent", ["All Opponents"] + opponents)
opp_filter = None if selected_opponent == "All Opponents" else selected_opponent

sportsbook_line = st.number_input("📈 Sportsbook Line (PTS)", 0.0, 60.0, 22.5)

# -------------------------------
# 🔮 Predict
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
