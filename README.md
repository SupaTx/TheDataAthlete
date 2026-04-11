# TheDataAthlete

![Banner](https://ik.imagekit.io/RealPaddi/Gemini_Generated_Image_mkqhexmkqhexmkqh%20(1).png)

> **LightGBM-powered football betting predictor** – Over/Under, 1X2, BTTS, and clean sheet probabilities from historical match data.

---

## 📌 Overview

**TheDataAthlete** is a machine learning project that builds a **unified betting model** for football matches. Using a single feature set (shots, corners, Elo ratings, rolling form, etc.), the model predicts:

- **Total expected goals** (regression) → derived Over/Under probabilities for 0.5, 1.5, 2.5, 3.5, 4.5 markets.
- **Binary outcomes**: 1X, X2, 12 (no draw), BTTS, Home Clean Sheet, Away Clean Sheet.

The model was trained on the **Club Football Match Data (2000–2025)** and **ELO ratings**, achieving reliable accuracy on test sets (2023+ matches). All code is provided as a Google Colab notebook, and trained models can be downloaded and used locally.

---

## ✨ Features

- **Single regression model** for total goals → all Over/Under thresholds derived via Poisson distribution.
- **Six binary LightGBM classifiers** for key betting markets.
- **Time‑based train/test split** (no look‑ahead bias).
- **Rolling feature engineering** – last‑5 goals scored/conceded per team.
- **Easy local inference** – load the saved `.pkl` models and run predictions on any match.
- **Visual outputs** – clear probability tables and the model’s best pick.

---

## 🧠 Model Training

### Data Sources

- `MATCHES.csv` – match statistics (goals, shots, corners, cards, form, results) from 2000 to 2025.
- `ELO.csv` – team Elo ratings over time (merged into matches).

### Feature Set (example)

- Basic stats: `HomeShots`, `AwayShots`, `HomeTarget`, `AwayTarget`, `HomeCorners`, `AwayCorners`, `HomeFouls`, `AwayFouls`, cards, reds.
- Form columns: `Form3Home`, `Form5Home`, `Form3Away`, `Form5Away`.
- Elo ratings: `HomeElo`, `AwayElo`.
- Rolling averages: `HomeGoalsScoredLast5`, `HomeGoalsConcededLast5`, `AwayGoalsScoredLast5`, `AwayGoalsConcededLast5`.

### Targets

- **Regression**: `TotalGoals` (FTHome + FTAway)
- **Binary**:
  - `1X` (Home win or draw)
  - `X2` (Draw or away win)
  - `12` (Home or away win – no draw)
  - `HomeCleanSheet` (Away goals = 0)
  - `AwayCleanSheet` (Home goals = 0)
  - `BTTS` (Both teams scored)

### Training Pipeline

All steps are in the included Colab notebook (`TheDataAthlete_Training.ipynb`):

1. Load and merge `MATCHES.csv` with `ELO.csv`.
2. Feature engineering (rolling averages, etc.).
3. Chronological split (train < 2023, test ≥ 2023).
4. Train LightGBM regression model for total goals.
5. Train six LightGBM binary classifiers.
6. Save models and feature list via `joblib`.
7. Evaluate on test set (MAE, AUC, accuracy).

---

## 📁 Project Structure

```
TheDataAthlete/
├── TheDataAthlete_Training.ipynb   # Full training pipeline (Google Colab)
├── models/                         # Trained models (downloaded from Drive)
│   ├── lgb_total_goals_model.pkl
│   ├── lgb_binary_models.pkl
│   └── feature_cols.pkl
├── assets/                         # Screenshots of winning predictions
│   ├── real_madrid_bayern.png
│   ├── barca_atletico.png
│   └── psg_liverpool.png
├── predict_match.py                # Local inference script
├── requirements.txt
└── README.md
```

---

## 🚀 Usage

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/TheDataAthlete.git
cd TheDataAthlete
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the trained models

The models are saved in Google Drive after training. Place the three `.pkl` files into the `models/` folder.

### 4. Run a prediction for a new match

Edit `predict_match.py` with your match’s statistics (shots, corners, Elo, etc.), then run:

```bash
python predict_match.py
```

Example output:

```
==================================================
MATCH: Barcelona vs Atlético Madrid
==================================================
Predicted total goals: 3.85

OVER/UNDER MARKETS (highest probability first):
  Over_0.5: 0.979 (97.9%)
  Over_1.5: 0.896 (89.6%)
  Over_2.5: 0.738 (73.8%)
  ...

BINARY MARKETS:
  1X: YES (0.752)
  12: YES (0.806)
  BTTS: YES (0.740)

🎯 MODEL'S BEST PICK: Over_0.5 with 97.9% probability
```

---

## 📊 Results

The model has been tested on real matches from 2023–2025. Example winning predictions (screenshots in `assets/`):

- **Real Madrid vs Bayern Munich** – Over 0.5 ✅ (97.9%)
- **Barcelona vs Atlético Madrid** – Over 0.5 ✅ (98.6%)
- **PSG vs Liverpool** – Over 0.5 ✅ (98.6%)

> **Always bet on the market with the highest probability** – the model is calibrated to maximize confidence in that single outcome.

---

## 🛠 Requirements

- Python 3.8+
- pandas
- numpy
- lightgbm
- scikit-learn
- scipy
- joblib

See `requirements.txt` for exact versions.

---

## 🔮 Future Improvements

- Add expected goals (xG) data from Understat or StatsBomb.
- Implement rolling win/draw/loss streaks as features.
- Build a simple web dashboard (Streamlit) to input match stats and get predictions.
- Backtest betting ROI using historical odds.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
---

## 🙏 Acknowledgments

- Data: [Club Football Match Data (2000–2025)](https://github.com/xgabora/Club-Football-Match-Data-2000-2025)
- LightGBM developers
- Poisson distribution methodology inspired by Dixon‑Coles model

---

**Made with ⚽ by TheDataAthlete**