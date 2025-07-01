import streamlit as st
import pandas as pd
import time
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, commonteamroster
import xgboost as xgb

st.set_page_config(page_title="NBA Edge Predictor", layout="centered")
st.title("🏀 NBA Edge Predictor")

# -------------------------------
# How this works
# -------------------------------
with st.expander("ℹ️ How this model works"):
    st.markdown("""
    - Pulls **2024–25 NBA season** game logs
    - Uses a trained **XGBoost model** to predict points
    - Features: MIN, FGA, FG3A, AST, REB, TOV, DaysRest, Home/Away
    - Compares model prediction to sportsbook line
    """)

# -------------------------------
# Load team > player list
# -------------------------------
@st.cache_data
def load_teams():
    return sorted(teams.get_teams(), key=lambda x: x['full_name'])

@st.cache_data
def load_all_players():
    return players.get_players()

teams_list = load_teams()
player_list = load_all_players()

team_names = [t['full_name'] for t in teams_list]
selected_team = st.selectbox("Select Team", team_names)

team_id = next(t['id'] for t in teams_list if t['full_name'] == selected_team)

# ✅ THIS IS YOUR ORIGINAL WORKING LOGIC
roster_df = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]
roster_names = roster_df['PLAYER'].tolist()
active_players = [p for p in player_list if p['is_active']]
team_players = [p for p in active_players if p['full_name'] in roster_names]

# Handle empty case
if not team_players:
    st.warning("No active players found.")
    st.stop()

player_names = sorted([p['full_name'] for p in team_players])
selected_player = st.selectbox("Select Player", player_names)
selected_player_id = next(p['id'] for p in team_players if p['full_name'] == selected_player)

# -------------------------------
# Game filter controls
# -------------------------------
max_games = st.slider("Number of recent games to include", 5, 30, 15)
sportsbook_line = st.number_input("📈 Sportsbook Line (PTS)", 0.0, 60.0, 22.5)

@st.cache_data
def get_player_gamelog(player_id):
    try:
        time.sleep(0.5)
        return playergamelog.PlayerGameLog(player_id=player_id, season='2024-25').get_data_frames()[0]
    except:
        return pd.DataFrame()

df_games = get_player_gamelog(selected_player_id)

# Extract opponent list
opponents = sorted(set(df_games['MATCHUP'].str.extract(r'@ (.*)|vs\. (.*)')[0].dropna()))
selected_opponent = st.selectbox("Optional: Filter by Opponent", ["All Opponents"] + opponents)
opp_filter = None if selected_opponent == "All Opponents" else selected_opponent

# -------------------------------
# Run model on Go button
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
    prediction = model.predict(pd.DataFrame([X.mean()]))[0]
    edge = prediction - sportsbook_line
    return prediction, edge

# -------------------------------
# Go button
# -------------------------------
if st.button("🚀 Go"):
    with st.spinner("Running prediction..."):
        df_filtered = df_games.copy()
        if opp_filter:
            df_filtered = df_filtered[df_filtered['MATCHUP'].str.contains(opp_filter)]
        df_filtered = df_filtered.sort_values('GAME_DATE', ascending=False).head(max_games)

        prediction, edge = train_model(df_filtered)

        if prediction is None:
            st.warning("Not enough data to train model.")
        else:
            st.markdown(f"### 🔮 Predicted Points: `{prediction:.2f}`")
            st.markdown(f"### 📊 Sportsbook Line: `{sportsbook_line}`")
            st.markdown(f"### 🧮 Edge: `{edge:.2f}`")

            if edge > 1:
                st.success("🔥 VALUE BET: Take the OVER")
            elif edge < -1:
                st.error("🧊 VALUE BET: Take the UNDER")
            else:
                st.info("❌ No clear edge.")
