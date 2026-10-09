from pathlib import Path
import pandas as pd
from collections import defaultdict, deque

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'elo_history.csv'
OUTPUT_PATH = BASE_DIR / 'features.csv'

df = pd.read_csv(DATA_PATH, parse_dates=['Date'])


last5 = defaultdict(lambda: deque(maxlen=5))

def form_points(dq):
    return sum(dq) if len(dq) > 0 else 0 

def form_avg(dq):
    return (sum(dq) / len(dq)) if len(dq) > 0 else 1.0  


h2h = defaultdict(lambda: deque(maxlen=5))  

rows = []
for _, r in df.iterrows():
    home, away, result = r['HomeTeam'], r['AwayTeam'], r['FTR']

    
    home_form = form_avg(last5[home])
    away_form = form_avg(last5[away])

  
    key = frozenset({home, away})
    h2h_deque = h2h[key]
    home_h2h_pts = sum(p for t, p in h2h_deque if t == home)
    away_h2h_pts = sum(p for t, p in h2h_deque if t == away)
    h2h_games = len(h2h_deque)

    rows.append({
        'Season': r['Season'], 'Date': r['Date'], 'HomeTeam': home, 'AwayTeam': away,
        'Elo_diff': r['Elo_home_before'] - r['Elo_away_before'],
        'Home_form': home_form, 'Away_form': away_form,
        'Form_diff': home_form - away_form,
        'H2H_home_pts': home_h2h_pts, 'H2H_away_pts': away_h2h_pts, 'H2H_games': h2h_games,
        'FTR': result,
    })

   
    home_pts = 3 if result == 'H' else (1 if result == 'D' else 0)
    away_pts = 3 if result == 'A' else (1 if result == 'D' else 0)
    last5[home].append(home_pts)
    last5[away].append(away_pts)
    h2h_deque.append((home, home_pts))
    h2h_deque.append((away, away_pts))

feat_df = pd.DataFrame(rows)
feat_df.to_csv(OUTPUT_PATH, index=False)
print(feat_df.shape)
print(feat_df.head(10))
print(feat_df['FTR'].value_counts())
