import streamlit as st
import pandas as pd
import time
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, commonteamroster
import xgboost
import xgboost as xgb

st.set_page_config(page_title="NBA Edge Predictor", layout="centered")
st.title("🏀 NBA Edge Predictor")

# -------------------------------
# ℹ️ How it works
# -------------------------------
with st.expander("ℹ️ How this model works"):
    st.markdown("""
    - Uses **2024–25 NBA regular season** game logs
    - Trains an **XGBoost regression model** using recent stats
    - Features: minutes, FGA, AST, REB, TOV, rest days
    - Optional filter by opponent
    - Compares model output to **sportsbook line** to find betting value
    """)

# -------------------------------
# Load Teams and Active Players
# -------------------------------
@st.cache_data
def get_teams():
    return sorted(teams.get_teams(), key=lambda x: x['full_name'])

@st.cache_data
def get_players():
    return players.get_players()

team_data = get_teams()
player_data = get_players()

team_names = [t['full_name'] for t in team_data]
selected_team = st.selectbox("Select a Team", team_names)

team_id = next(t['id'] for t in team_data if t['full_name'] == selected_team)
roster = commonteamroster.CommonTeamRoster(team_id=team_id)
roster_df = roster.get_data_frames()[0]
roster_names = roster_df['PLAYER'].tolist()

# Match full player info from active player list
active_players = [p for p in player_data if p['is_active']]
team_players = [p for p in active_players if p['full_name'] in roster_names]
player_names = sorted([p['full_name'] for p in team_players])
selected_player = st.selectbox("Select a Player", player_names)
selected_player_id = next(p['id'] for p in team_players if p['full_name'] == selected_player)

# -------------------------------
# Game filter options
# -------------------------------
max_games = st.slider("How many recent games to use?", 5, 30, 15)
sportsbook_line = st.number_input("📈 Sportsbook Line (PTS)", 0.0, 60.0, 22.5)

@st.cache_data
def get_player_game_data(player_id):
    try:
        time.sleep(0.5)
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season='2024-25')
        df = gamelog.get_data_frames()[0]
        return df
    except:
        return pd.DataFrame()

df_games = get_player_game_data(selected_player_id)
opponents = sorted(set(df_games['MATCHUP'].str.extract(r'@ (.*)|vs\. (.*)')[0].dropna()))
selected_opponent = st.selectbox("Optional: Filter by Opponent", ["All Opponents"] + opponents)
opponent_filter = None if selected_opponent == "All Opponents" else selected_opponent

# -------------------------------
# Train and Predict — Only After "Go"
# -------------------------------
def prepare_data(df):
    df = df.rename(columns={'MATCHUP': 'Opponent', 'MIN': 'Minutes'})
    df['Home'] = df['Opponent'].apply(lambda x: 1 if 'vs.' in x else 0)
    df['DaysRest'] = (
        pd.to_datetime(df['GAME_DATE']) - pd.to_datetime(df['GAME_DATE']).shift(-1)
    ).dt.days.fillna(2)
    df = df[['Minutes', 'FGA', 'FG3A', 'AST', 'REB', 'TOV', 'Home', 'DaysRest', 'PTS']]
    return df.dropna()

def train_model(df):
    df = prepare_data(df)
    if len(df) < 5:
        return None, None
    X = df.drop(columns='PTS')
    y = df['PTS']
    model = xgb.XGBRegressor()
    model.fit(X, y)
    avg_input = pd.DataFrame([X.mean()])
    return model, avg_input

# -------------------------------
# GO Button
# -------------------------------
if st.button("🚀 Go"):
    with st.spinner("Training model and predicting..."):
        filtered_df = df_games.copy()
        if opponent_filter:
            filtered_df = filtered_df[filtered_df['MATCHUP'].str.contains(opponent_filter)]

        filtered_df = filtered_df.sort_values('GAME_DATE', ascending=False).head(max_games)
        model, avg_input = train_model(filtered_df)

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
            st.warning("Not enough game data to train the model.")
