# LaLiga Elo Ratings & Match Outcome Prediction

An end-to-end data science project built on 30 years of LaLiga match results: build an **Elo rating system from scratch**, turn it into **pre-match features**, and train classifiers to predict **Home win / Draw / Away win** — validated with a **time-aware backtest** against a naive baseline.

> **Status:** Work in progress. Core pipeline (Elo → features → models → backtest) is complete. See the [Roadmap](#roadmap) and [Changelog](#changelog) for what's next.

---

## Highlights

- Custom Elo implementation, validated by checking that it recovers known football history (Barcelona's 2009–13 peak, Atlético's rise under Simeone, Villarreal's mid-2000s era).
- **No data leakage:** every feature for a match is computed using only matches played *before* it.
- **Time-aware split:** trained on 26 seasons, tested on the 5 most recent — no shuffling.
- Beats the "always predict home win" baseline on both accuracy and log-loss.

---

## Dataset

| Property | Value |
|---|---|
| File | `LaLiga_Matches.csv` |
| Matches | 11,664 |
| Seasons | 31 (1995–96 to 2025–26) |
| Teams | 48 |
| Columns | `Season, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG, HTAG, HTR` |
| Missing values | 2 rows with missing half-time data (HTHG / HTAG / HTR) |

Class balance (full-time result): Home 5,500 · Away 3,172 · Draw 2,992.

This is match-level data only — no shots, xG, odds, or player stats — so all signal has to be engineered from results and goals.

---

## Method

### 1. Elo ratings (`step1_elo.py`)

Matches are processed in strict chronological order, keeping **one rating per team** in a dictionary that persists across the whole history.

- Initial rating: **1500** for every team
- K-factor: **20**
- Home advantage: **+100** added to the home team's rating *only* when computing expected score
- Expected score: `E_A = 1 / (1 + 10^((R_B - R_A) / 400))`
- Update: `R_new = R + K * (S - E)` with S = 1 / 0.5 / 0 for win / draw / loss

Sanity check: Elo is zero-sum, so the sum of final ratings equals `1500 × 48 = 72,000` ✔

![Elo trajectories](elo_trajectories.png)

### 2. Feature engineering (`step2_features.py`)

All features are recorded **before** the match result updates any state.

| Feature | Description |
|---|---|
| `Elo_diff` | Home Elo − Away Elo, before the match |
| `Home_form`, `Away_form` | Average points per game over each team's last 5 matches (neutral prior of 1.0 if no history) |
| `Form_diff` | `Home_form − Away_form` |
| `H2H_home_pts`, `H2H_away_pts` | Points each side earned in the last 5 meetings between these two teams |
| `H2H_games` | Number of prior meetings available (max 5) |

### 3. Modeling & backtest (`step3_model.py`)

- **Train:** 1995–96 → 2020–21 (26 seasons)
- **Test:** 2021–22 → 2025–26 (5 seasons, 1,620 matches)
- **Models:** Multinomial Logistic Regression (standardized features), Random Forest (300 trees, depth 6, min leaf 20)
- **Metrics:** accuracy and log-loss
- **Baseline:** always predict a home win (log-loss baseline uses training-set class frequencies as probabilities)

---

## Results

| Model | Accuracy | Log-loss |
|---|---|---|
| Baseline (always Home) | 44.9% | 1.0707 |
| Logistic Regression | **52.9%** | **0.9842** |
| Random Forest | 51.9% | 0.9921 |

**Confusion matrix — Logistic Regression** (rows = actual, columns = predicted; order H, D, A)

|  | Pred H | Pred D | Pred A |
|---|---|---|---|
| **Actual H** | 631 | 0 | 97 |
| **Actual D** | 310 | 0 | 121 |
| **Actual A** | 235 | 0 | 226 |

**Random Forest feature importances**

| Feature | Importance |
|---|---|
| `Elo_diff` | 0.590 |
| `Form_diff` | 0.125 |
| `Home_form` | 0.080 |
| `Away_form` | 0.070 |
| `H2H_away_pts` | 0.066 |
| `H2H_home_pts` | 0.058 |
| `H2H_games` | 0.011 |

Elo difference carries most of the predictive signal.

---

## Known limitations

- **Draws are never predicted.** Neither model ever outputs a draw as its top class (recall = 0 for draws). Draws are ~26% of matches but are rarely the single most probable outcome, so argmax never selects them. This is a well-known difficulty in football prediction; addressing it is the next planned improvement.
- **Elo constants are not tuned.** K = 20 and home advantage = 100 were chosen as reasonable defaults, not optimized.
- **Results-only data.** No xG, shots, lineups, injuries, or odds, which caps achievable accuracy.
- **Promoted/relegated teams** enter at 1500 regardless of strength, which can distort early ratings for new clubs.

---

## Project structure

```
.
├── LaLiga_Matches.csv      # raw data
├── step1_elo.py            # compute chronological Elo ratings -> elo_history.csv
├── step2_features.py       # build pre-match features        -> features.csv
├── step3_model.py          # train, evaluate, backtest models
├── elo_history.csv         # per-match Elo before/after
├── features.csv            # model-ready feature table
├── elo_trajectories.png    # Elo trajectory plot (top clubs)
└── README.md
```

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib

python step1_elo.py
python step2_features.py
python step3_model.py
```

Run the scripts in order — each one reads the output of the previous step. Results are deterministic.

---

## Roadmap

- [x] Chronological Elo system with per-team ratings
- [x] Validate Elo against known football history
- [x] Feature engineering (Elo diff, rolling form, head-to-head)
- [x] Time-aware backtest vs. baseline
- [ ] Address draw prediction (two-stage draw / not-draw model or probability thresholds)
- [ ] Tune K-factor and home advantage on a validation split
- [ ] Add goal-difference-based form features
- [ ] Try gradient boosting (XGBoost / LightGBM)
- [ ] Calibration analysis (do predicted probabilities match observed frequencies?)
- [ ] Walk-forward validation across multiple seasons
- [ ] Interactive dashboard / visualizations
- [ ] Final write-up

---

## Changelog

### 2026-10-09
- Initial pipeline: Elo ratings, feature engineering, Logistic Regression and Random Forest models
- Time-aware backtest on 2021–22 → 2025–26 (LogReg: 52.9% accuracy, log-loss 0.984 vs baseline 44.9% / 1.071)
- Identified draw-blindness as the main limitation

---

## Lessons learned

- Elo needs **one rating per team carried across the whole history**; a spreadsheet column referencing "the row above" can't do this, which is why the implementation lives in Python with a dictionary.
- Random train/test splits leak the future into the past on time-ordered data; use a chronological split.
- Accuracy alone is misleading for probabilistic outcomes — log-loss and a sensible baseline matter.
