from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'features.csv'

df = pd.read_csv(DATA_PATH, parse_dates=['Date'])

FEATURES = ['Elo_diff', 'Form_diff', 'Home_form', 'Away_form',
            'H2H_home_pts', 'H2H_away_pts', 'H2H_games']
TARGET = 'FTR'

# --- Time-aware split: last 5 seasons = test, everything before = train ---
seasons_sorted = sorted(df['Season'].unique())
test_seasons = seasons_sorted[-5:]
train_seasons = seasons_sorted[:-5]

print("Train seasons:", train_seasons[0], "to", train_seasons[-1], f"({len(train_seasons)} seasons)")
print("Test seasons:", test_seasons)

train = df[df['Season'].isin(train_seasons)].copy()
test = df[df['Season'].isin(test_seasons)].copy()

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Baseline: always predict home win ---
baseline_pred = ['H'] * len(y_test)
baseline_acc = accuracy_score(y_test, baseline_pred)

# --- Baseline log-loss: predict train-set class frequencies for every row ---
class_order = sorted(y_train.unique())
train_freqs = y_train.value_counts(normalize=True)[class_order].values
baseline_probs = np.tile(train_freqs, (len(y_test), 1))
baseline_logloss = log_loss(y_test, baseline_probs, labels=class_order)

print(f"\n--- Baseline (always predict Home Win) ---")
print(f"Accuracy: {baseline_acc:.4f}")
print(f"Log-loss (using train class frequencies as probs): {baseline_logloss:.4f}")

# --- Model 1: Multinomial Logistic Regression ---
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_s, y_train)
lr_pred = logreg.predict(X_test_s)
lr_probs = logreg.predict_proba(X_test_s)
lr_acc = accuracy_score(y_test, lr_pred)
lr_logloss = log_loss(y_test, lr_probs, labels=logreg.classes_)

print(f"\n--- Logistic Regression ---")
print(f"Accuracy: {lr_acc:.4f}")
print(f"Log-loss: {lr_logloss:.4f}")
print(confusion_matrix(y_test, lr_pred, labels=['H', 'D', 'A']))
print(classification_report(y_test, lr_pred))

# --- Model 2: Random Forest ---
rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42)
rf.fit(X_train, y_train)  # tree models don't need scaling
rf_pred = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_logloss = log_loss(y_test, rf_probs, labels=rf.classes_)

print(f"\n--- Random Forest ---")
print(f"Accuracy: {rf_acc:.4f}")
print(f"Log-loss: {rf_logloss:.4f}")
print(confusion_matrix(y_test, rf_pred, labels=['H', 'D', 'A']))
print(classification_report(y_test, rf_pred))

print("\n--- Feature importances (Random Forest) ---")
for f, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"{f}: {imp:.4f}")

# --- Summary table ---
print("\n=== SUMMARY ===")
print(f"{'Model':<20}{'Accuracy':<12}{'Log-loss':<12}")
print(f"{'Baseline (Home)':<20}{baseline_acc:<12.4f}{baseline_logloss:<12.4f}")
print(f"{'Logistic Regression':<20}{lr_acc:<12.4f}{lr_logloss:<12.4f}")
print(f"{'Random Forest':<20}{rf_acc:<12.4f}{rf_logloss:<12.4f}")
