import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import commonteamroster, playergamelog
import xgboost as xgb

st.set_page_config(page_title="NBA Edge Predictor", layout="centered")
st.title("🏀 NBA Edge Predictor")

# -------------------------------
# Load team and player dropdowns
# -------------------------------
team_data = teams.get_teams()
player_data = players.get_players()

team_names = sorted([team['full_name'] for team in team_data])
selected_team = st.selectbox("Select Team", team_names)

team_id = next(team['id'] for team in team_data if team['full_name'] == selected_team)
roster_df = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]
roster_names = roster_df['PLAYER'].tolist()

active_players = [p for p in player_data if p['is_active']]
team_players = [p for p in active_players if p['full_name'] in roster_names]

if not team_players:
    st.warning("No active players found.")
    st.stop()

player_names = sorted([p['full_name'] for p in team_players])
selected_player = st.selectbox("Select Player", player_names)
selected_player_id = next(p['id'] for p in team_players if p['full_name'] == selected_player)

# Input line
sportsbook_line = st.number_input("📈 Sportsbook Line (PTS)", 0.0, 60.0, 22.5)
max_games = 15

# -------------------------------
# Model and helper functions
# -------------------------------
def get_recent_gamelog(player_id, limit=15):
    try:
        time.sleep(0.5)
        df = playergamelog.PlayerGameLog(player_id=player_id, season='2024-25').get_data_frames()[0]
        return df.sort_values("GAME_DATE", ascending=False).head(limit)
    except:
        return pd.DataFrame()

def prepare_data(df):
    df = df.rename(columns={'MATCHUP': 'Opponent', 'MIN': 'Minutes'})
    df['Home'] = df['Opponent'].apply(lambda x: 1 if 'vs.' in x else 0)
    df['DaysRest'] = (
        pd.to_datetime(df['GAME_DATE']) - pd.to_datetime(df['GAME_DATE']).shift(-1)
    ).dt.days.fillna(2)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    return df[['GAME_DATE', 'Minutes', 'FGA', 'FG3A', 'AST', 'REB', 'TOV', 'Home', 'DaysRest', 'PTS']].dropna()

def train_model(df, sportsbook_line):
    df = prepare_data(df)
    if len(df) < 5:
        return None, None, None
    X = df.drop(columns=['PTS', 'GAME_DATE'])
    y = df['PTS']
    model = xgb.XGBRegressor()
    model.fit(X, y)
    avg_input = pd.DataFrame([X.mean()])
    prediction = model.predict(avg_input)[0]
    edge = prediction - sportsbook_line
    return prediction, edge, df

# -------------------------------
# GO Button
# -------------------------------
if st.button("🚀 Go"):
    with st.spinner("Training model and generating prediction..."):
        df_raw = get_recent_gamelog(selected_player_id, limit=max_games)
        prediction, edge, df = train_model(df_raw, sportsbook_line)

        if prediction is None:
            st.warning("Not enough game data to build the model.")
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

            # -------------------------------
            # 📈 Graph 1: Points vs Sportsbook Line
            # -------------------------------
            st.subheader("📈 Recent Points vs Sportsbook Line")
            fig1, ax1 = plt.subplots()
            ax1.plot(df['GAME_DATE'], df['PTS'], marker='o', label='Points')
            ax1.axhline(sportsbook_line, linestyle='--', color='red', label='Line')
            ax1.set_ylabel("Points")
            ax1.set_xlabel("Game Date")
            ax1.set_title(f"{selected_player} – Last {len(df)} Games")
            ax1.grid(True)
            ax1.legend()
            ax1.set_xticks(df['GAME_DATE'])
            ax1.set_xticklabels(df['GAME_DATE'].dt.strftime('%b %d'), rotation=45, ha='right')
            fig1.tight_layout()
            st.pyplot(fig1)

            # -------------------------------
            # 🧠 Graph 2: Minutes vs Points
            # -------------------------------
            st.subheader("🧠 Minutes vs Points Correlation")
            fig2, ax2 = plt.subplots()
            ax2.scatter(df['Minutes'], df['PTS'], c='steelblue', edgecolors='black', s=100)
            m, b = np.polyfit(df['Minutes'], df['PTS'], 1)
            ax2.plot(df['Minutes'], m * df['Minutes'] + b, '--', color='black', label=f"y={m:.2f}x+{b:.1f}")
            ax2.set_xlabel("Minutes")
            ax2.set_ylabel("Points")
            ax2.grid(True)
            ax2.legend()
            fig2.tight_layout()
            st.pyplot(fig2)

            # -------------------------------
            # 📉 Graph 3: Points Minus Line
            # -------------------------------
            st.subheader("📉 Game-by-Game Edge vs Line")
            fig3, ax3 = plt.subplots()
            errors = df['PTS'] - sportsbook_line
            labels = df['GAME_DATE'].dt.strftime('%b %d')
            tick_interval = max(1, len(labels) // 8)
            tick_positions = np.arange(len(labels))[::tick_interval]
            ax3.bar(labels, errors, color=['green' if val > 0 else 'red' for val in errors])
            ax3.axhline(0, linestyle='--', color='black')
            ax3.set_ylabel("Points - Line")
            ax3.set_title("Edge vs Line by Game")
            ax3.set_xticks(tick_positions)
            ax3.set_xticklabels([labels[i] for i in tick_positions], rotation=45, ha='right')
            fig3.tight_layout()
            st.pyplot(fig3)
