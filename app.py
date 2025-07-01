import streamlit as st
import pandas as pd
import time
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, commonteamroster
import xgboost as xgb

st.set_page_config(page_title="NBA Edge Predictor", layout="centered")
st.title("🏀 NBA Edge Predictor")

# -------------------------------
# Load teams and players
# -------------------------------
@st.cache_data
def load_teams_and_players():
    return teams.get_teams(), players.get_players()

team_data, player_data = load_teams_and_players()
team_names = sorted([t['full_name'] for t in team_data])
selected_team = st.selectbox("Select Team", team_names)

# Get team ID and roster
team_id = next(t['id'] for t in team_data if t['full_name'] == selected_team)
roster_df = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]
roster_names = roster_df['PLAYER'].tolist()

# Match with active players
active_players = [p for p in player_data if p['is_active']]
team_players = [p for p in active_players if p['full_name'] in roster_names]

if not team_players:
    st.warning("No active players found.")
    st.stop()

player_names = sorted([p['full_name'] for p in team_players])
selected_player = st.selectbox("Select Player", player_names)
selected_player_id = next(p['id'] for p in team_players if p['full_name'] == selected_player)

# -------------------------------
# Sportsbook line input
# -------------------------------
sportsbook_line = st.number_input("📈 Sportsbook Line (PTS)", 0.0, 60.0, 22.5)
max_games = 15

# -------------------------------
# Fetch game log (2024–25 only)
# -------------------------------
def get_gamelog(player_id):
    try:
        time.sleep(0.5)
        return playergamelog.PlayerGameLog(player_id=player_id, season='2024-25').get_data_frames()[0]
    except:
        return pd.DataFrame()

def prepare_data(df):
    df = df.rename(columns={'MATCHUP': 'Opponent', 'MIN': 'Minutes'})
    df['Home'] = df['Opponent'].apply(lambda x: 1 if 'vs.' in x else 0)
    df['DaysRest'] = (
        pd.to_datetime(df['GAME_DATE']) - pd.to_datetime(df['GAME_DATE']).shift(-1)
    ).dt.days.fillna(2)
    return df[['Minutes', 'FGA', 'FG3A', 'AST', 'REB', 'TOV', 'Home', 'DaysRest', 'PTS']].dropna()

def train_model(df):
    df = prepare_data(df)
    if len(df) < 5:
        return None, None
    X = df.drop(columns='PTS')
    y = df['PTS']
    model = xgb.XGBRegressor()
    model.fit(X, y)
    avg_input = pd.DataFrame([X.mean()])
    prediction = model.predict(avg_input)[0]
    edge = prediction - sportsbook_line
    return prediction, edge

# -------------------------------
# GO Button triggers everything
# -------------------------------
if st.button("🚀 Go"):
    with st.spinner("Fetching data and training model..."):
        df = get_gamelog(selected_player_id)
        df = df.sort_values("GAME_DATE", ascending=False).head(max_games)
        prediction, edge = train_model(df)

        if prediction is None:
            st.warning("Not enough data to train the model.")
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
