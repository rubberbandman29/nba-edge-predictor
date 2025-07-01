import streamlit as st
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import commonteamroster

st.set_page_config(page_title="NBA Debugger", layout="centered")
st.title("🧪 NBA Roster Debugger")

# Load teams and players
st.header("1. Load Teams")
teams_list = teams.get_teams()
team_names = sorted([t['full_name'] for t in teams_list])
selected_team = st.selectbox("Select Team", team_names)

team_id = next(t['id'] for t in teams_list if t['full_name'] == selected_team)
st.write(f"Selected Team ID: `{team_id}`")

# Try getting roster
st.header("2. Load Team Roster via commonteamroster")
try:
    roster_df = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]
    st.dataframe(roster_df[['PLAYER', 'PLAYER_ID']])
    roster_names = roster_df['PLAYER'].tolist()
except Exception as e:
    st.error(f"Failed to load roster: {e}")
    st.stop()

# Try loading all players
st.header("3. Load All Active Players")
try:
    all_players = players.get_players()
    active_players = [p for p in all_players if p['is_active']]
    st.write(f"✅ Loaded {len(active_players)} active players")
except Exception as e:
    st.error(f"Failed to load players: {e}")
    st.stop()

# Match full_name
st.header("4. Match Active Players to Team Roster")
team_players = [p for p in active_players if p['full_name'] in roster_names]

if not team_players:
    st.warning("⚠️ No players matched. Showing debug info:")
    st.text("First 5 roster names:")
    st.write(roster_names[:5])
    st.text("First 5 active player names:")
    st.write([p['full_name'] for p in active_players[:5]])
else:
    st.success(f"✅ Matched {len(team_players)} players")
    player_names = sorted([p['full_name'] for p in team_players])
    selected_player = st.selectbox("Select Player", player_names)
