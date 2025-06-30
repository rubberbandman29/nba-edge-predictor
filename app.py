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
# 🔍 HOW THIS MODEL WORKS
# ---------------------------
with st.expander("ℹ️ How this model works"):
    st.markdown("""
    - 📦 **Data Source:** Uses `nba_api` to pull player game logs for the selected season.
    - 🧠 **Model Type:** Trains an **XGBoost regression model** on each player's past performance.
    - 🔢 **Features Used:**
      - Minutes played
      - Field goal attempts (FGA) and 3-point attempts (FG3A)
      - Assists, rebounds, turnovers
      - Home/Away indicator
      - Days of rest since previous game
    - 📈 **Training:** The model learns how these features affect point totals.
    - 🔮 **Prediction:** It uses **average stats** from the season to predict the next game's point total.
    - ⚖️ **Betting Edge:** Compares predicted total vs. sportsbook line to suggest:
      - 🔥 Over (if edge > +1)
      - 🧊 Under (if edge < -1)
      - ❌ No edge (if within ±1)
    """)

# ---------------------------
# 📁 Load team and player data
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
# 📌 Get player ID and data
# ---------------------------
def get_player_id(name):
    result = players.find_players_by_full_name(name)
    return result[0]['id'] if result else None

@st.cache_resource
def train_player_model(player_id):
    try:
        time.sleep(0.6)  # avoid nba_api rate limits
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season='2024-25')
        df = gamelog.get_data_frames()[0]
        if df.empty:
            return None, None

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
    except:
        return None, None

# ---------------------------
# 🔮 Predict and compare
# ---------------------------
if selected_player:
    player_id = get_player_id(selected_player)
    if player_id:
        sportsbook_line = st.number_input("📊 Enter Sportsbook Line (PTS)", 0.0, 60.0, 22.5)

        if st.button("Predict"):
            with st.spinner("Training model and making prediction..."):
                model, avg_input = train_player_model(player_id)
                if model:
                    predicted_pts = model.predict(avg_input)[0]
                    edge = predicted_pts - sportsbook_line

                    st.markdown(f"### 🔮 Predicted Points: `{predicted_pts:.2f}`")
                    st.markdown(f"### 📊 Sportsbook Line: `{sportsbook_line}`")
                    st.markdown(f"### 🧮 Edge: `{edge:.2f}`")

                    if edge > 1:
                        st.success("🔥 VALUE BET: Take the OVER")
                    elif edge < -1:
                        st.error("🧊 VALUE BET: Take the UNDER")
                    else:
                        st.info("❌ No clear edge.")
                else:
                    st.warning("Not enough data to train model. Try a different player.")
