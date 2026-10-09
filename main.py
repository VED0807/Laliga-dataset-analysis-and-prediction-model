import pandas as pd

df = pd.read_csv('LaLiga_Matches.csv')

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)


INITIAL_RATING = 1500
K = 20
HOME_ADV = 100  

ratings = {}

def get_rating(team):
    return ratings.get(team, INITIAL_RATING)

def expected_score(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))


def actual_score(result, side):
    if result == 'D':
        return 0.5
    if side == 'home':
        return 1.0 if result == 'H' else 0.0
    return 1.0 if result == 'A' else 0.0


history = []

for idx, row in df.iterrows():
    home, away, result = row['HomeTeam'], row['AwayTeam'], row['FTR']

    r_home = get_rating(home)
    r_away = get_rating(away)

    e_home = expected_score(r_home + HOME_ADV, r_away)
    e_away = 1 - e_home

    s_home = actual_score(result, 'home')
    s_away = actual_score(result, 'away')

    new_r_home = r_home + K * (s_home - e_home)
    new_r_away = r_away + K * (s_away - e_away)

    ratings[home] = new_r_home
    ratings[away] = new_r_away

    history.append({
        'Date': row['Date'], 'Home': home, 'Away': away,
        'R_home_before': r_home, 'R_away_before': r_away,
        'E_home': e_home, 'FTR': result,
        'R_home_after': new_r_home, 'R_away_after': new_r_away
    })

history_df = pd.DataFrame(history)
print(history_df.head(10))
print(ratings)  

print(sum(ratings.values()), len(ratings))

import matplotlib.pyplot as plt

def get_team_trajectory(history_df, team):
    """Extract a team's rating after each of their matches, whether home or away."""
    home_matches = history_df[history_df['Home'] == team][['Date', 'R_home_after']].rename(
        columns={'R_home_after': 'Rating'})
    away_matches = history_df[history_df['Away'] == team][['Date', 'R_away_after']].rename(
        columns={'R_away_after': 'Rating'})
    combined = pd.concat([home_matches, away_matches]).sort_values('Date')
    return combined

teams_to_plot = ['Real Madrid', 'Barcelona', 'Ath Madrid', 'Villarreal']

plt.figure(figsize=(14, 7))
for team in teams_to_plot:
    traj = get_team_trajectory(history_df, team)
    plt.plot(traj['Date'], traj['Rating'], label=team, linewidth=1.2)

plt.axhline(1500, color='gray', linestyle='--', linewidth=0.8, label='Starting rating')
plt.title('Elo Rating Trajectories — LaLiga (1995–2025)')
plt.xlabel('Date')
plt.ylabel('Elo Rating')
plt.legend()
plt.tight_layout()
plt.savefig('elo_trajectories.png', dpi=150)
plt.show()