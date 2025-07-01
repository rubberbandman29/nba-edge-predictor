import streamlit as st
import pandas as pd
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, commonteamroster
import xgboost as xgb
import time

st.set_page_config(page_title="NBA Edge Predictor", layout="centered")
st.title("🏀 NBA Edge Predictor")

with st.expander("ℹ️ How this model works"):
    st.markdown("""
    - Uses **2024–25 season** stats only
    - Trains a custom **XGBoost regression model**
    - Based on minutes, FGA, 3PA, AST, REB, TOV, rest days
    - Compares model prediction to sportsbook line
    - Optional: Filter by opponent
    """)

# -------------------------------
# Load Teams and Rosters
# -------------------------------
@st.cache_data
def load_teams():
    return sorted(teams.get_teams(), key=lambda x: x['full_name'])

@st.cache_data
def load_players():
    return players.get_players()

teams_list = load_teams()
all_players = load_players()

team_names = [t['full_name'] for t in teams_list]
selected_team = st.selectbox("Select Team", team_names)

team_id = next(t['id'] for t in teams_list if t['full_name'] == selected_team)
roster_df = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]
roster_names = roster_df['PLAYER'].tolist()

# Match roster to active players with IDs
active_players = [p for p in all_players if p['is_active']]
team_players = [p for p in active_players if p['full_name'] in roster_names]

if not team_players:
    st.warning("No active players found for this team.")
    st.stop()

player_names = sorted([p['full_name'] for p in team_players])
selected_player = st.selectbox("Select Player", player_names)
selected_player_id = next(p['id'] for p in team_players if p['full_name'] == selected_player)

# -------------------------------
# Game & Opponent Filters (UI only, no model runs yet)
# -------------------------------
max_games = st.slider("How many recent games to include?", 5, 30, 15)
sportsbook_line = st.number_input("📈 Sportsbook Line (PTS)", 0.0, 60.0, 22.5)

@st.cache_data
def load_gamelog(player_id):
    try:
        time.sleep(0.5)
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season='2024-25')
        return gamelog.get_data_frames()[0]
    except:
        return pd.DataFrame()

gamelog_df = load_gamelog(selected_player_id)

opponents = sorted(set(gamelog_df['MATCHUP'].str.extract(r'@ (.*)|vs\. (.*)')[0].dropna()))
selected_opponent = st.selectbox("Optional: Filter by Opponent", ["All Opponents"] + opponents)
opp_filter = None if selected_opponent == "All Opponents" else selected_opponent

# -------------------------------
# Run Model AFTER Go Button Click
# -------------------------------
def prepare_data(df):
    df = df.rename(columns={'MATCHUP': 'Opponent', 'MIN': 'Minutes'})
    df['Home'] = df['Opponent'].apply(lambda x: 1 if 'vs.' in x else 0)
    df['DaysRest'] = (
        pd.to_datetime(df['GAME_DATE']) - pd.to_datetime(df['GAME_DATE']).shift(-1)
    ).dt.days.fillna(2)
    df = df[['Minutes', 'FGA', 'FG3A', 'AST', 'REB', 'TOV', 'Home', 'DaysRest', 'PTS']]
    return df.dropna()

def train_and_predict(df, sportsbook_line):
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
# 🚀 GO Button
# -------------------------------
if st.button("🚀 Go"):
    with st.spinner("Training model and making prediction..."):
        df_filtered = gamelog_df.copy()
        if opp_filter:
            df_filtered = df_filtered[df_filtered['MATCHUP'].str.contains(opp_filter)]
        df_filtered = df_filtered.sort_values('GAME_DATE', ascending=False).head(max_games)

        prediction, edge = train_and_predict(df_filtered, sportsbook_line)

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
