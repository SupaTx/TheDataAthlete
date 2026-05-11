# predict_match.py

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# Style config
# ─────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linestyle':   '--',
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'text.color':       '#e0e0e0',
    'axes.labelcolor':  '#e0e0e0',
    'xtick.color':      '#a0a0a0',
    'ytick.color':      '#a0a0a0',
    'axes.edgecolor':   '#2a2d3a',
})

HOME_COLOR  = '#4A90D9'
AWAY_COLOR  = '#E8654A'
OVER_COLOR  = '#4CAF7D'
UNDER_COLOR = '#555e6e'
BEST_COLOR  = '#9C6FDE'

# ─────────────────────────────────────────
# 1. Load models
# ─────────────────────────────────────────
model_goals   = joblib.load('.\\models\\lgb_total_goals_model.pkl')
models_binary = joblib.load('.\\models\\lgb_binary_models.pkl')
feature_cols  = joblib.load('.\\models\\feature_cols.pkl')

# ─────────────────────────────────────────
# 2. Over/Under helper
# ─────────────────────────────────────────
def predict_over_under(expected_goals: float) -> dict:
    thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]
    probs = {}
    for t in thresholds:
        k = int(np.floor(t))
        prob_over = 1 - poisson.cdf(k, expected_goals)
        probs[f'Over_{t}']  = prob_over
        probs[f'Under_{t}'] = 1 - prob_over
    return probs

# ─────────────────────────────────────────
# 3. Match input  ← edit these per match
# ─────────────────────────────────────────
HOME_TEAM = 'Tottenham Hotspur'
AWAY_TEAM = 'Leeds United'

match_data = {
    'HomeShots':              11,  # From previous H2H encounter
    'AwayShots':              13,  # From previous H2H encounter
    'HomeTarget':              4,  # From previous H2H encounter
    'AwayTarget':              4,  # From previous H2H encounter
    'HomeCorners':             5,  # Estimated based on team averages
    'AwayCorners':             4,  # Estimated based on team averages
    'HomeFouls':               9,  # From previous H2H encounter
    'AwayFouls':              12,  # From previous H2H encounter
    'HomeYellow':              2,  # From previous H2H encounter
    'AwayYellow':              3,  # From previous H2H encounter
    'HomeRed':                 0,  # From previous H2H encounter
    'AwayRed':                 0,  # From previous H2H encounter
    'Form3Home':               7,  # Points from last 3: W-W-D
    'Form5Home':               8,  # Points from last 5: W-W-D-L-L
    'Form3Away':               7,  # Points from last 3: W-D-W
    'Form5Away':              11,  # Points from last 5: W-D-W-W-D
    'HomeElo':              1771,  # [ClubElo](http://clubelo.com/Tottenham)
    'AwayElo':              1799,  # [ClubElo](http://clubelo.com/Leeds)
    'HomeGoalsScoredLast5':  1.2,  #
    'AwayGoalsScoredLast5':  2.0,  #
    'HomeGoalsConcededLast5':1.6,  #
    'AwayGoalsConcededLast5':1.0,  #
}


new_match = pd.DataFrame([match_data])[feature_cols]

# ─────────────────────────────────────────
# 4. Run predictions
# ─────────────────────────────────────────
expected_goals = model_goals.predict(new_match)[0]
ou_probs       = predict_over_under(expected_goals)

binary_results = {
    name: model_bin.predict_proba(new_match)[0, 1]
    for name, model_bin in models_binary.items()
}

all_probs   = {**ou_probs, **binary_results}
best_market = max(all_probs, key=all_probs.get)
best_prob   = all_probs[best_market]

# ─────────────────────────────────────────
# 5. Build summary DataFrames
# ─────────────────────────────────────────
ou_df = (
    pd.DataFrame(list(ou_probs.items()), columns=['Market', 'Probability'])
      .assign(
          Type       = lambda d: d['Market'].str.split('_').str[0],
          Threshold  = lambda d: d['Market'].str.split('_').str[1],
          Pct_Label  = lambda d: (d['Probability'] * 100).round(1).astype(str) + '%',
      )
      .sort_values('Probability', ascending=False)
      .reset_index(drop=True)
)

binary_df = (
    pd.DataFrame(list(binary_results.items()), columns=['Market', 'Probability'])
      .assign(
          Decision  = lambda d: np.where(d['Probability'] >= 0.5, 'YES', 'NO'),
          Pct_Label = lambda d: (d['Probability'] * 100).round(1).astype(str) + '%',
      )
      .sort_values('Probability', ascending=False)
      .reset_index(drop=True)
)

stats_df = pd.DataFrame({
    'Stat':  ['Shots', 'On Target', 'Corners', 'Fouls', 'Yellow Cards', 'Red Cards',
              'Form (last 3)', 'Form (last 5)', 'Goals Scored L5', 'Goals Conceded L5', 'ELO'],
    HOME_TEAM: [match_data['HomeShots'], match_data['HomeTarget'], match_data['HomeCorners'],
                match_data['HomeFouls'], match_data['HomeYellow'], match_data['HomeRed'],
                match_data['Form3Home'], match_data['Form5Home'],
                match_data['HomeGoalsScoredLast5'], match_data['HomeGoalsConcededLast5'],
                match_data['HomeElo']],
    AWAY_TEAM: [match_data['AwayShots'], match_data['AwayTarget'], match_data['AwayCorners'],
                match_data['AwayFouls'], match_data['AwayYellow'], match_data['AwayRed'],
                match_data['Form3Away'], match_data['Form5Away'],
                match_data['AwayGoalsScoredLast5'], match_data['AwayGoalsConcededLast5'],
                match_data['AwayElo']],
})

# ─────────────────────────────────────────
# 6. Visualise
# ─────────────────────────────────────────
fig = plt.figure(figsize=(18, 22))
fig.patch.set_facecolor('#0f1117')

gs = gridspec.GridSpec(
    4, 2,
    figure=fig,
    hspace=0.45,
    wspace=0.35,
    top=0.91, bottom=0.04,
    left=0.07, right=0.97,
)

# ── Title banner ──────────────────────────
ax_title = fig.add_axes([0, 0.93, 1, 0.07])
ax_title.set_facecolor('#161926')
ax_title.axis('off')
ax_title.text(0.5, 0.72, f'{HOME_TEAM}  vs  {AWAY_TEAM}',
              ha='center', va='center', fontsize=22, fontweight='bold', color='white',
              transform=ax_title.transAxes)
ax_title.text(0.5, 0.22, f'Expected Total Goals: {expected_goals:.2f}   |   '
              f'Best Pick: {best_market.replace("_", " ")} @ {best_prob:.1%}',
              ha='center', va='center', fontsize=12, color='#a0a0b0',
              transform=ax_title.transAxes)

# ── (A) Over/Under horizontal bar chart ──
ax_ou = fig.add_subplot(gs[0, :])
ax_ou.set_facecolor('#1a1d27')

over_df  = ou_df[ou_df['Type'] == 'Over'].sort_values('Threshold')
under_df = ou_df[ou_df['Type'] == 'Under'].sort_values('Threshold')

x      = np.arange(len(over_df))
width  = 0.38

bars_over  = ax_ou.bar(x - width/2, over_df['Probability'],  width, color=OVER_COLOR,  alpha=0.88, label='Over',  zorder=3)
bars_under = ax_ou.bar(x + width/2, under_df['Probability'], width, color=UNDER_COLOR, alpha=0.88, label='Under', zorder=3)

ax_ou.set_xticks(x)
ax_ou.set_xticklabels([f'{t} Goals' for t in over_df['Threshold']], fontsize=11)
ax_ou.set_ylabel('Probability', fontsize=11)
ax_ou.set_ylim(0, 1.12)
ax_ou.set_title('Over / Under Market Probabilities', fontsize=13, fontweight='bold',
                color='white', pad=10)
ax_ou.axhline(0.5, color='white', linestyle='--', linewidth=0.8, alpha=0.4, zorder=2)
ax_ou.legend(framealpha=0.15, fontsize=10)

for bar in bars_over:
    h = bar.get_height()
    ax_ou.text(bar.get_x() + bar.get_width()/2, h + 0.015,
               f'{h:.1%}', ha='center', va='bottom', fontsize=9, color=OVER_COLOR, fontweight='bold')
for bar in bars_under:
    h = bar.get_height()
    ax_ou.text(bar.get_x() + bar.get_width()/2, h + 0.015,
               f'{h:.1%}', ha='center', va='bottom', fontsize=9, color='#9aa0ab', fontweight='bold')

# ── (B) Binary markets horizontal bars ───
ax_bin = fig.add_subplot(gs[1, 0])
ax_bin.set_facecolor('#1a1d27')

colors_bin = [OVER_COLOR if p >= 0.5 else UNDER_COLOR for p in binary_df['Probability']]
bars_bin   = ax_bin.barh(binary_df['Market'], binary_df['Probability'],
                         color=colors_bin, alpha=0.88, zorder=3)
ax_bin.set_xlim(0, 1.15)
ax_bin.axvline(0.5, color='white', linestyle='--', linewidth=0.8, alpha=0.4)
ax_bin.set_xlabel('Probability', fontsize=10)
ax_bin.set_title('Binary Markets', fontsize=13, fontweight='bold', color='white', pad=10)
ax_bin.invert_yaxis()

for bar, (_, row) in zip(bars_bin, binary_df.iterrows()):
    w = bar.get_width()
    badge_color = OVER_COLOR if row['Decision'] == 'YES' else '#c0392b'
    ax_bin.text(w + 0.025, bar.get_y() + bar.get_height()/2,
                f"{row['Pct_Label']}  [{row['Decision']}]",
                va='center', fontsize=9,
                color=badge_color, fontweight='bold')

# ── (C) Poisson goal distribution ────────
ax_poi = fig.add_subplot(gs[1, 1])
ax_poi.set_facecolor('#1a1d27')

goals_range = np.arange(0, 10)
pmf_vals    = poisson.pmf(goals_range, expected_goals)
bar_colors  = [BEST_COLOR if g == round(expected_goals) else HOME_COLOR for g in goals_range]

ax_poi.bar(goals_range, pmf_vals, color=bar_colors, alpha=0.85, zorder=3, width=0.65)
ax_poi.set_xlabel('Total Goals', fontsize=10)
ax_poi.set_ylabel('Probability', fontsize=10)
ax_poi.set_title(f'Poisson Goal Distribution  (λ = {expected_goals:.2f})',
                 fontsize=13, fontweight='bold', color='white', pad=10)
ax_poi.set_xticks(goals_range)

for i, (g, p) in enumerate(zip(goals_range, pmf_vals)):
    if p > 0.01:
        ax_poi.text(g, p + 0.003, f'{p:.2%}', ha='center', fontsize=8, color='#a0a0b0')

# ── (D) Head-to-head stat comparison ─────
ax_stats = fig.add_subplot(gs[2, :])
ax_stats.set_facecolor('#1a1d27')

stat_labels = stats_df['Stat'].tolist()
home_vals   = stats_df[HOME_TEAM].tolist()
away_vals   = stats_df[AWAY_TEAM].tolist()

x2    = np.arange(len(stat_labels))
w2    = 0.38

ax_stats.bar(x2 - w2/2, home_vals, w2, color=HOME_COLOR, alpha=0.85,
             label=HOME_TEAM, zorder=3)
ax_stats.bar(x2 + w2/2, away_vals, w2, color=AWAY_COLOR, alpha=0.85,
             label=AWAY_TEAM, zorder=3)

ax_stats.set_xticks(x2)
ax_stats.set_xticklabels(stat_labels, rotation=30, ha='right', fontsize=10)
ax_stats.set_ylabel('Value', fontsize=11)
ax_stats.set_title('Head-to-Head Match Stats', fontsize=13, fontweight='bold',
                   color='white', pad=10)
ax_stats.legend(framealpha=0.15, fontsize=10)

# ── (E) Summary table ─────────────────────
ax_tbl = fig.add_subplot(gs[3, 0])
ax_tbl.set_facecolor('#1a1d27')
ax_tbl.axis('off')
ax_tbl.set_title('Binary Markets — Decision Table', fontsize=13,
                 fontweight='bold', color='white', pad=10)

tbl_data   = binary_df[['Market', 'Pct_Label', 'Decision']].values.tolist()
col_labels = ['Market', 'Probability', 'Decision']

tbl = ax_tbl.table(
    cellText    = tbl_data,
    colLabels   = col_labels,
    cellLoc     = 'center',
    loc         = 'center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.6)

for (row, col), cell in tbl.get_celld().items():
    cell.set_facecolor('#1a1d27' if row > 0 else '#2a2d3a')
    cell.set_edgecolor('#2a2d3a')
    cell.set_text_props(color='white')
    if row > 0 and col == 2:
        txt = tbl_data[row - 1][2]
        cell.set_text_props(
            color=OVER_COLOR if txt == 'YES' else '#c0392b',
            fontweight='bold'
        )

# ── (F) Best pick highlight ───────────────
ax_best = fig.add_subplot(gs[3, 1])
ax_best.set_facecolor('#1a1d27')
ax_best.axis('off')

rect = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                      boxstyle='round,pad=0.05',
                      linewidth=2, edgecolor=BEST_COLOR,
                      facecolor='#1e1630', transform=ax_best.transAxes)
ax_best.add_patch(rect)
ax_best.text(0.5, 0.78, 'MODEL\'S BEST PICK',
             ha='center', va='center', fontsize=11, color='#a090d0',
             fontweight='bold', transform=ax_best.transAxes)
ax_best.text(0.5, 0.52, best_market.replace('_', ' '),
             ha='center', va='center', fontsize=20, color='white',
             fontweight='bold', transform=ax_best.transAxes)
ax_best.text(0.5, 0.28, f'{best_prob:.1%} probability',
             ha='center', va='center', fontsize=15, color=BEST_COLOR,
             fontweight='bold', transform=ax_best.transAxes)

plt.savefig('match_prediction_report.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print(f"Report saved → match_prediction_report.png")