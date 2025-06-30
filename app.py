import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# ----------- Get player ID from name -----------
def get_player_id(player_name):
    try:
        return players.find_players_by_full_name(player_name)[0]['id']
    except IndexError:
        return None

# ----------- Load Game Log & Train Model -----------
def train_model(player_id):
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season='2024-25')
        df = gamelog.get_data_frames()[0]
        df = df.rename(columns={'MATCHUP': 'Opponent', 'MIN': 'Minutes'})
        df['Home'] = df['Opponent'].apply(lambda x: 1 if 'vs.' in x else 0)
        df['DaysRest'] = (pd.to_datetime(df['GAME_DATE']) - pd.to_datetime(df['GAME_DATE']).shift(-1)).dt.days.fillna(2)
        df = df[['Minutes', 'FGA', 'FG3A', 'AST', 'REB', 'TOV', 'Home', 'DaysRest', 'PTS']].dropna()
        X = df.drop(columns='PTS')
        y = df['PTS']
        model = xgb.XGBRegressor()
        model.fit(X, y)
        return model
    except:
        return None

# ----------- Streamlit App UI -----------
st.title("🏀 NBA Edge Predictor")
st.markdown("Predict player performance and identify value vs the sportsbook line.")

player_name = st.text_input("Enter player name:", "LaMelo Ball")

player_id = get_player_id(player_name)

if player_id:
    model = train_model(player_id)
    if model:
        st.subheader("📋 Input Expected Game Stats")

        col1, col2 = st.columns(2)
        with col1:
            minutes = st.slider("Minutes", 10, 48, 35)
            fga = st.slider("Field Goal Attempts (FGA)", 0, 30, 18)
            fg3a = st.slider("Three Point Attempts (FG3A)", 0, 15, 6)
            tov = st.slider("Turnovers (TOV)", 0, 10, 3)
        with col2:
            ast = st.slider("Assists (AST)", 0, 15, 7)
            reb = st.slider("Rebounds (REB)", 0, 20, 6)
            home = st.radio("Home Game?", ["Yes", "No"]) == "Yes"
            days_rest = st.slider("Days Rest", 0, 5, 2)

        sportsbook_line = st.number_input("📊 Sportsbook Line (Points)", 0.0, 60.0, 22.5)

        # Prediction
        if st.button("Predict"):
            input_df = pd.DataFrame([{
                'Minutes': minutes,
                'FGA': fga,
                'FG3A': fg3a,
                'AST': ast,
                'REB': reb,
                'TOV': tov,
                'Home': 1 if home else 0,
                'DaysRest': days_rest
            }])
            prediction = model.predict(input_df)[0]
            edge = prediction - sportsbook_line

            st.markdown(f"### 🔮 Predicted Points: `{prediction:.2f}`")
            st.markdown(f"### 📊 Sportsbook Line: `{sportsbook_line}`")
            st.markdown(f"### 🧮 Edge: `{edge:.2f}`")

            if edge > 1:
                st.success("🔥 VALUE BET: Take the OVER")
            elif edge < -1:
                st.error("🧊 VALUE BET: Take the UNDER")
            else:
                st.info("❌ No clear edge. Consider skipping this one.")
    else:
        st.warning("Could not load player game log. Try a different player.")
else:
    st.warning("Player not found.")
