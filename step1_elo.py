from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'LaLiga_Matches.csv'
OUTPUT_PATH = BASE_DIR / 'elo_history.csv'

df = pd.read_csv(DATA_PATH)
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
    if side == 'home':
        return 1.0 if result == 'H' else (0.5 if result == 'D' else 0.0)
    else:
        return 1.0 if result == 'A' else (0.5 if result == 'D' else 0.0)

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
        'Season': row['Season'], 'Date': row['Date'], 'HomeTeam': home, 'AwayTeam': away,
        'FTR': result,
        'Elo_home_before': r_home, 'Elo_away_before': r_away,
        'Elo_home_after': new_r_home, 'Elo_away_after': new_r_away,
    })

history_df = pd.DataFrame(history)
history_df.to_csv(OUTPUT_PATH, index=False)
print(history_df.shape)
print(history_df.head())
print("Sum of final ratings:", sum(ratings.values()), "num teams:", len(ratings))
