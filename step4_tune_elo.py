
import itertools
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

INITIAL_RATING = 1500


df = pd.read_csv('LaLiga_Matches.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)

seasons = sorted(df['Season'].unique())
test_seasons = seasons[-5:]
val_seasons = seasons[-10:-5]
train_seasons = seasons[:-10]
print(f"Train: {train_seasons[0]}..{train_seasons[-1]} ({len(train_seasons)})")
print(f"Val  : {val_seasons[0]}..{val_seasons[-1]} ({len(val_seasons)})")
print(f"Test : {test_seasons[0]}..{test_seasons[-1]} ({len(test_seasons)})")

home = df['HomeTeam'].to_numpy()
away = df['AwayTeam'].to_numpy()
res = df['FTR'].to_numpy()
season_arr = df['Season'].to_numpy()
n = len(df)


last5 = defaultdict(lambda: deque(maxlen=5))
h2h = defaultdict(lambda: deque(maxlen=5))
other = np.zeros((n, 6))  

for i in range(n):
    h, a, r = home[i], away[i], res[i]
    hf = (sum(last5[h]) / len(last5[h])) if last5[h] else 1.0
    af = (sum(last5[a]) / len(last5[a])) if last5[a] else 1.0
    dq = h2h[frozenset((h, a))]
    other[i] = [hf, af, hf - af,
                sum(p for t, p in dq if t == h),
                sum(p for t, p in dq if t == a),
                len(dq)]
    hp = 3 if r == 'H' else (1 if r == 'D' else 0)
    ap = 3 if r == 'A' else (1 if r == 'D' else 0)
    last5[h].append(hp)
    last5[a].append(ap)
    dq.append((h, hp))
    dq.append((a, ap))



def elo_diff_series(k, home_adv):
    """Pre-match Elo_home - Elo_away for every match, chronological, leak-free."""
    ratings = {}
    out = np.empty(n)
    for i in range(n):
        h, a, r = home[i], away[i], res[i]
        rh = ratings.get(h, INITIAL_RATING)
        ra = ratings.get(a, INITIAL_RATING)
        out[i] = rh - ra
        e_h = 1.0 / (1.0 + 10 ** ((ra - (rh + home_adv)) / 400.0))
        s_h = 1.0 if r == 'H' else (0.5 if r == 'D' else 0.0)
        ratings[h] = rh + k * (s_h - e_h)
        ratings[a] = ra + k * ((1.0 - s_h) - (1.0 - e_h))
    return out


def evaluate(k, home_adv, fit_seasons, eval_seasons):
    X = np.column_stack([elo_diff_series(k, home_adv), other])
    fit_mask = np.isin(season_arr, fit_seasons)
    eval_mask = np.isin(season_arr, eval_seasons)
    scaler = StandardScaler().fit(X[fit_mask])
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X[fit_mask]), res[fit_mask])
    probs = model.predict_proba(scaler.transform(X[eval_mask]))
    preds = model.classes_[probs.argmax(axis=1)]
    return (log_loss(res[eval_mask], probs, labels=model.classes_),
            accuracy_score(res[eval_mask], preds))



K_GRID = [10, 15, 20, 25, 30, 40, 50]
HA_GRID = [0, 25, 50, 75, 100, 125, 150]

rows = []
for k, ha in itertools.product(K_GRID, HA_GRID):
    ll, acc = evaluate(k, ha, train_seasons, val_seasons)
    rows.append({'K': k, 'HomeAdv': ha, 'val_logloss': ll, 'val_acc': acc})

grid = pd.DataFrame(rows).sort_values('val_logloss').reset_index(drop=True)
grid.to_csv('tuning_grid_results.csv', index=False)

print("\nTop 10 by validation log-loss:")
print(grid.head(10).to_string(index=False))

best = grid.iloc[0]
best_k, best_ha = int(best['K']), int(best['HomeAdv'])
old_ll, old_acc = grid[(grid.K == 20) & (grid.HomeAdv == 100)][['val_logloss', 'val_acc']].iloc[0]
print(f"\nBest on validation: K={best_k}, HomeAdv={best_ha}  "
      f"(val log-loss {best['val_logloss']:.4f}, acc {best['val_acc']:.4f})")
print(f"Old defaults K=20, HomeAdv=100 on validation: log-loss {old_ll:.4f}, acc {old_acc:.4f}")


fit_for_test = seasons[:-5] 
tl_old, ta_old = evaluate(20, 100, fit_for_test, test_seasons)
tl_new, ta_new = evaluate(best_k, best_ha, fit_for_test, test_seasons)

print("\n=== TEST SET (2021-22 .. 2025-26), evaluated once ===")
print(f"{'Config':<28}{'Accuracy':<12}{'Log-loss':<12}")
print(f"{'Old  K=20, HomeAdv=100':<28}{ta_old:<12.4f}{tl_old:<12.4f}")
print(f"{f'Tuned K={best_k}, HomeAdv={best_ha}':<28}{ta_new:<12.4f}{tl_new:<12.4f}")

pd.DataFrame([
    {'config': 'old K=20 HA=100', 'test_acc': ta_old, 'test_logloss': tl_old},
    {'config': f'tuned K={best_k} HA={best_ha}', 'test_acc': ta_new, 'test_logloss': tl_new},
]).to_csv('tuning_test_comparison.csv', index=False)


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pivot = grid.pivot(index='K', columns='HomeAdv', values='val_logloss').sort_index()
fig, ax = plt.subplots(figsize=(8, 5.5))
im = ax.imshow(pivot.values, cmap='viridis_r', aspect='auto')
ax.set_xticks(range(len(pivot.columns)), pivot.columns)
ax.set_yticks(range(len(pivot.index)), pivot.index)
ax.set_xlabel('Home advantage (Elo points)')
ax.set_ylabel('K-factor')
ax.set_title('Validation log-loss by Elo constants (lower is better)')
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha='center', va='center', color='white', fontsize=7)
bi = list(pivot.index).index(best_k)
bj = list(pivot.columns).index(best_ha)
ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=2))
fig.colorbar(im, ax=ax, label='log-loss')
plt.tight_layout()
plt.savefig('tuning_heatmap.png', dpi=150)
print("\nSaved tuning_grid_results.csv, tuning_test_comparison.csv, tuning_heatmap.png")
