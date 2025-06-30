import streamlit as st
import pandas as pd
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# ---------- Cache player/team data ----------
@st.cache_data
def get_teams():
    return sorted(teams.get_teams(), key=lambda x: x['full_name'])

@st.cache_data
def get_players_by_team(team_id):
    all_players = players.get_active_players()
    # NBA API doesn't link players to teams directly, use basic mapping
    team_players = [p for p in all_players if team_id in p.get('team_id', '')]
    return team_players

@st.cache_data
def get_all_active_players():
    return players.get_active_players()

# ---------- Get player ID ----------
def get_player_id_by_name(name):
    player_list = players.find_players_by_full_name(name)
    return player_list[0]['id'] if player_list else None

# ---------- Train model ----------
@st.cache_resource
def train_model(player_id):
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season='2024-25')
        df = gamelog.get_data_frames()[0]
        df = df.rename(columns={'MATCHUP': 'Opponent', 'MIN': 'Minutes'})
        df['Home'] = df['Opponent'].apply(lambda x: 1 if 'vs.' in x else 0)
        df['DaysRest'] = (pd.to_datetime(df['GAME_DATE']) - pd.to_datetime(df['GAME_DATE']).shift(-1)).dt.days.fillna(2)

        df = df[['Minutes', 'FGA', 'FG3A', 'AST', 'REB', 'TOV', 'Home', 'DaysRest', 'PTS']].dropna()

        if df.shape[0] < 5:
            return None, None

        X = df.drop(columns='PTS')
        y = df['PTS']
        model = xgb.XGBRegressor()
        model.fit(X, y)
        avg_input = pd.DataFrame([X.mean()])
        return model, avg_input
    except:
        return None, None

# ---------- Streamlit UI ----------
st.set_page_config(page_title="NBA Edge Predictor", layout="centered")
st.title("🏀 NBA Edge Predictor")
st.caption("Select a player and compare model prediction vs sportsbook line.")

# ---------- TEAM + PLAYER SELECT ----------
team_list = get_teams()
team_names = [t['full_name'] for t in team_list]
selected_team = st.selectbox("Select Team", team_names)

# Filter players
all_players = get_all_active_players()
team_id = [t['id'] for t in team_list if t['full_name'] == selected_team][0]
team_players = [p['full_name'] for p in all_players if p.get('team_id') == team_id]
selected_player = st.selectbox("Select Player", sorted(team_players))

# ---------- Prediction ----------
if selected_player:
    player_id = get_player_id_by_name(selected_player)
    model, avg_input = train_model(player_id)

    if model is not None:
        sportsbook_line = st.number_input("📊 Enter Sportsbook Line (PTS)", 0.0, 60.0, 22.5)

        if st.button("Predict"):
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
                st.info("❌ No clear edge. Skip this one.")
    else:
        st.warning("Insufficient game data to train model. Try another player.")
