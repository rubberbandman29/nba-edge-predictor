import streamlit as st
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import commonteamroster

st.set_page_config(page_title="NBA Edge Predictor", layout="centered")
st.title("🏀 NBA Edge Predictor")

# Load teams and players once
team_data = teams.get_teams()
player_data = players.get_players()

# Team dropdown
team_names = sorted([t['full_name'] for t in team_data])
selected_team = st.selectbox("Select Team", team_names)
team_id = next(t['id'] for t in team_data if t['full_name'] == selected_team)

# Load team roster (live from API)
roster_df = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]
roster_names = roster_df['PLAYER'].tolist()

# Match to active player list
active_players = [p for p in player_data if p['is_active']]
team_players = [p for p in active_players if p['full_name'] in roster_names]

# Player dropdown
if not team_players:
    st.warning("No players found for this team.")
    st.stop()

player_names = sorted([p['full_name'] for p in team_players])
selected_player = st.selectbox("Select Player", player_names)
