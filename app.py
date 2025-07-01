import streamlit as st
import pandas as pd
import time
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import playergamelog
from sklearn.metrics import mean_squared_error
import xgboost as xgb

st.set_page_config(page_title="NBA Edge Predictor", layout="centered")
st.title("🏀 NBA Edge Predictor")

# -------------------------------
# ℹ️ How the Model Works
# -------------------------------
with st.expander("ℹ️ How this model works"):
    st.markdown("""
    - 🔁 Pulls up to 30 most recent games from **2022–25 seasons**
    - 🎯 Optional filter for a specific **opponent**
    - 🧠 Trains a custom **XGBoost regression model**
    - 📈 Predicts **next-game points** using average stats
    - ⚖️ Compares to sportsbook line to show value bet
    """)

# -------------------------------
# 🏀 Load Teams and Players
# -------------------------------
@st.cache_data
def get_all_teams():
    return sorted(teams.get_teams(), key=lambda x: x['full_name'])

@st.cache_data
def get_all_players():
    return players.get_active_players()

team_list = get_all_teams()
player_list = get_all_players()

team_names = [t['full_name'] for t in team_list]
selected_team = st.selectbox("Select Team", team_names)

team_abbr = next((t['abbreviation'] for t in team_list if t['full_name'] == selected_team), None)

# -------------------------------
# 🔍 Filter Players by Recent Game Logs
# -------------------------------
@st.cache_data
def filter_players_by_team(team_abbr):
    matched_players = []
    for p in player_list:
        try:
            time.sleep(0.2)
            gamelog = playergamelog.PlayerGameLog(player_id=p['id'], season='2024-25')
            df = gamelog.get_data_frames()[0]
            if not df.empty and any(team_abbr in s for s in df['MATCHUP'].values):
                matched_players.append(p['full_name'])
        except:
            continue
    return sorted(set(matched_players))

with st.spinner("Loading players..."):
    filtered_players = filter_players_by_team(team_abbr)

selected_player = st.selectbox("Select Player", filtered_players)

# -------------------------------
# 🧠 Get Player Data and Train Model
# -------------------------------
def get_player_id(name):
    result = players.find_players_by_full_name(name)
    return result[0]['id'] if result else None

def get_player_games(player_id, max_games=30, opponent=None):
    seasons = ['2024-25', '2023-24', '2022-23']
    dfs = []
    for season in seasons:
        try:
            time.sleep(0.6)
            gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
            df = gamelog.get_data_frames()[0]
            if not df.empty:
                df['Season'] = season
                dfs.append(df)
        except:
            continue
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs).sort_values('GAME_DATE', ascending=False)
    if opponent:
        df = df[df['MATCHUP'].str.contains(opponent)]
    return df.head(max_games)

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
# 🎛️ User Controls
# -------------------------------
if selected_player:
    player_id = get_player_id(selected_player)
    st.subheader("📊 Game Filters")
    max_games = st.slider("Number of recent games to include", 5, 30, 15)

    full_df = get_player_games(player_id, max_games=100)
    all_opponents = sorted(set(full_df['MATCHUP'].str.extract(r'@ (.*)|vs\. (.*)')[0].dropna()))
    selected_opp = st.selectbox("Optional: Filter by Opponent", ["All Opponents"] + all_opponents)
    opp_filter = None if selected_opp == "All Opponents" else selected_opp

    sportsbook_line = st.number_input("📈 Sportsbook Line (PTS)", 0.0, 60.0, 22.5)

    if st.button("Predict"):
        with st.spinner("Training model and predicting..."):
            df = get_player_games(player_id, max_games=max_games, opponent=opp_filter)
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
                st.warning("Not enough data to train the model.")
