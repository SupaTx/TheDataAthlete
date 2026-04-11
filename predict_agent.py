# predict_match.py

import pandas as pd
import numpy as np
import joblib
from scipy.stats import poisson

# ------------------------------
# 1. Load the models (update paths)
# ------------------------------
model_goals = joblib.load('lgb_total_goals_model.pkl')
models_binary = joblib.load('lgb_binary_models.pkl')
feature_cols = joblib.load('feature_cols.pkl')

# ------------------------------
# 2. Define a function to predict Over/Under from total goals
# ------------------------------
def predict_over_under(expected_goals):
    thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]
    probs = {}
    for t in thresholds:
        k = int(np.floor(t))
        prob_over = 1 - poisson.cdf(k, expected_goals)
        probs[f'Over_{t}'] = prob_over
        probs[f'Under_{t}'] = 1 - prob_over
    return probs

# ------------------------------
# 3. Define your match stats (example: Real Madrid vs Bayern)
#    You will replace these values for each new match
# ------------------------------
match_data = {
    'HomeShots': 16,
    'AwayShots': 15,
    'HomeTarget': 7,
    'AwayTarget': 8,
    'HomeCorners': 6,
    'AwayCorners': 6,
    'HomeFouls': 10,
    'AwayFouls': 11,
    'HomeYellow': 2,
    'AwayYellow': 2,
    'HomeRed': 0,
    'AwayRed': 0,
    'Form3Home': 6,
    'Form5Home': 12,
    'Form3Away': 7,
    'Form5Away': 13,
    'HomeElo': 1952,
    'AwayElo': 1996,
    'HomeGoalsScoredLast5': 2.2,
    'AwayGoalsScoredLast5': 2.8,
    'HomeGoalsConcededLast5': 1.0,
    'AwayGoalsConcededLast5': 0.8
}

# Convert to DataFrame and ensure correct column order
new_match = pd.DataFrame([match_data])
new_match = new_match[feature_cols]   # reorder columns exactly as training

# ------------------------------
# 4. Run predictions
# ------------------------------
expected_goals = model_goals.predict(new_match)[0]
ou_probs = predict_over_under(expected_goals)

# Binary markets
binary_results = {}
for name, model_bin in models_binary.items():
    prob = model_bin.predict_proba(new_match)[0, 1]
    binary_results[name] = prob

# ------------------------------
# 5. Display results clearly
# ------------------------------
print("=" * 50)
print(f"MATCH: {match_data.get('HomeTeam', 'Home')} vs {match_data.get('AwayTeam', 'Away')}")
print("=" * 50)
print(f"Predicted total goals: {expected_goals:.2f}\n")

print("OVER/UNDER MARKETS (highest probability first):")
# Sort by probability descending to see the best pick
sorted_ou = sorted(ou_probs.items(), key=lambda x: x[1], reverse=True)
for market, prob in sorted_ou:
    print(f"  {market}: {prob:.3f} ({prob*100:.1f}%)")

print("\nBINARY MARKETS (YES if probability >= 0.5):")
for name, prob in binary_results.items():
    decision = "YES" if prob >= 0.5 else "NO"
    print(f"  {name}: {decision} (prob: {prob:.3f})")

# ------------------------------
# 6. Best pick recommendation (highest probability overall)
# ------------------------------
all_probs = {**ou_probs, **binary_results}
best_market = max(all_probs, key=all_probs.get)
best_prob = all_probs[best_market]
print("\n" + "=" * 50)
print(f"🎯 MODEL'S BEST PICK: {best_market} with {best_prob:.1%} probability")
print("=" * 50)