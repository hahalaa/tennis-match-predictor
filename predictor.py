import joblib  # Standard for saving ML models
from pathlib import Path
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Machine Learning Imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Constants
START_YEAR = 2014
END_YEAR = 2024
ACCURACY_PLOT = "accuracy_comparison.png"
FEATURE_IMPORTANCE_PLOT = "feature_importance.png"
DEFAULT_RANK = 2000
DEFAULT_WIN_PCT = 0.5
VALID_SURFACES = {"Hard", "Clay", "Grass"}

# The exact list of features used for training and prediction
MODEL_FEATURES = [
    'p1_rank', 'p2_rank', 
    'p1_age', 'p2_age', 
    'p1_surface_win_pct', 'p2_surface_win_pct', 
    'h2h_diff'
]

# ==========================================
# 1. DATA LOADING
# ==========================================
def load_atp_data(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Downloads ATP match data from Jeff Sackmann's GitHub repository
    for a given range of years as CSV files (inclusive).

    Returns a single pandas DataFrame containing all matches.
    """
    BASE_URL = (
        "https://raw.githubusercontent.com/"
        "JeffSackmann/tennis_atp/master/atp_matches_{}.csv"
    )

    yearly_dfs = []
    print(f"⬇️  Downloading ATP data from {start_year} to {end_year}...")

    for year in range(start_year, end_year + 1):
        try:
            url = BASE_URL.format(year)
            df = pd.read_csv(url, on_bad_lines="skip")
            df["year"] = year
            yearly_dfs.append(df)

            print(f"   ✓ Loaded {year}: {len(df)} matches")

        except Exception as err:
            print(f"   ✗ Failed to load {year}: {err}")

    return pd.concat(yearly_dfs, ignore_index=True)

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw ATP match data by:
    - Handling missing values
    - Randomizing player order to create a balanced
      Player 1 vs Player 2 dataset
    - Creating a binary target (1 = P1 win, 0 = P1 loss)
    """
    df = df.copy()

    # Handle missing values
    df["winner_rank"] = df["winner_rank"].fillna(DEFAULT_RANK)
    df["loser_rank"]  = df["loser_rank"].fillna(DEFAULT_RANK)

    df["winner_age"] = df["winner_age"].fillna(df["winner_age"].median())
    df["loser_age"]  = df["loser_age"].fillna(df["loser_age"].median())

    # Randomise player order
    rng = np.random.default_rng(seed=42)
    swap_players = rng.random(len(df)) > DEFAULT_WIN_PCT

    # Build Player 1 / Player 2 dataset
    new_df = pd.DataFrame({
        "tourney_date": df["tourney_date"],
        "surface": df["surface"],
        "tourney_level": df["tourney_level"],

        "p1_name": np.where(swap_players, df["loser_name"], df["winner_name"]),
        "p1_rank": np.where(swap_players, df["loser_rank"], df["winner_rank"]),
        "p1_age":  np.where(swap_players, df["loser_age"],  df["winner_age"]),

        "p2_name": np.where(swap_players, df["winner_name"], df["loser_name"]),
        "p2_rank": np.where(swap_players, df["winner_rank"], df["loser_rank"]),
        "p2_age":  np.where(swap_players, df["winner_age"],  df["loser_age"]),
    })

    # Target: did Player 1 win?
    new_df["target"] = (~swap_players).astype(int)

    return new_df

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def add_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """
    Engineer time-aware features using only past match data:
    - Surface-specific win percentage
    - Head-to-head win differential (relative to Player 1)
    """
    print("⚙️  Engineering features...")

    df = df.copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df.sort_values("tourney_date").reset_index(drop=True)

    surface_history = {} # { 'Player': { 'Hard': [Wins, Total] } }
    h2h_history = {}     # { tuple('P1', 'P2'): [P1_wins, P2_wins] }

    p1_surface_pct = []
    p2_surface_pct = []
    h2h_diff = []

    def surface_win_pct(player, surface):
        if surface == "Unknown":
            return DEFAULT_WIN_PCT

        if player in surface_history and surface in surface_history[player]:
            wins, total = surface_history[player][surface]
            if total > 0:
                return wins / total
            else:
                return DEFAULT_WIN_PCT
            
        return DEFAULT_WIN_PCT

    def update_surface(player, surface, won):
        if player not in surface_history:
            surface_history[player] = {}
        if surface not in surface_history[player]:
            surface_history[player][surface] = [0, 0]
        surface_history[player][surface][1] += 1
        if won:
            surface_history[player][surface][0] += 1

    for _, row in df.iterrows():
        p1 = row["p1_name"]
        p2 = row["p2_name"]
        surface = row["surface"] if pd.notna(row["surface"]) else "Unknown"
        p1_won = row["target"] == 1

        # Surface features
        p1_surface_pct.append(surface_win_pct(p1, surface))
        p2_surface_pct.append(surface_win_pct(p2, surface))

        # H2H feature
        pair = tuple(sorted([p1, p2]))

        if pair in h2h_history:
            wins_a, wins_b = h2h_history[pair]

            if p1 == pair[0]:
                diff = wins_a - wins_b
            else:
                diff = wins_b - wins_a
        else:
            diff = 0

        h2h_diff.append(diff)

        # Update player surface histories
        update_surface(p1, surface, p1_won)
        update_surface(p2, surface, not p1_won)

        if pair not in h2h_history:
            h2h_history[pair] = [0, 0]

        # Compute winner index
        if (p1_won and p1 == pair[0]) or (not p1_won and p1 != pair[0]):
            winner_index = 0
        else:
            winner_index = 1

        h2h_history[pair][winner_index] += 1

    # Attach features
    df["p1_surface_win_pct"] = p1_surface_pct
    df["p2_surface_win_pct"] = p2_surface_pct
    df["h2h_diff"] = h2h_diff

    return df, surface_history, h2h_history

# ==========================================
# 4. MODEL TRAINING
# ==========================================
def train_and_evaluate(df):
    """
    Train multiple models and evaluate on the test year.
    Returns the best-performing model based on test accuracy.
    """
    print("🧠 Training models...")
    
    # Split by Year (Train: <2025, Test: 2025)
    train_mask = df['tourney_date'].dt.year < END_YEAR
    test_mask = df['tourney_date'].dt.year == END_YEAR
    
    X_train = df.loc[train_mask, MODEL_FEATURES]
    y_train = df.loc[train_mask, 'target']
    X_test  = df.loc[test_mask, MODEL_FEATURES]
    y_test  = df.loc[test_mask, 'target']
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }
    
    results = {}
    best_model = None
    best_acc = 0
    
    # Fit model to data
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = acc
        print(f"   - {name}: {acc:.4f}")
        
        if acc > best_acc:
            best_model = model
            best_acc = acc

    # Save Accuracy Plot
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
    plt.title(f"Model Accuracy (Test Year: {END_YEAR})")
    plt.ylim(DEFAULT_WIN_PCT, 0.75)
    plt.savefig(ACCURACY_PLOT)
    plt.close()
    print("   [Saved accuracy_comparison.png]")
    
    return best_model

def plot_feature_importance(model):
    """
    Plot and display feature importance and save it to a PNG file.
    """
    if not hasattr(model, "feature_importances_"):
        print("⚠️ Selected model does not support feature importance.")
        return

    importances = model.feature_importances_
    df_imp = pd.DataFrame({'Feature': MODEL_FEATURES, 'Importance': importances}).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_imp, hue='Feature', legend=False, palette='viridis')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PLOT)
    plt.close()
    
    print("\n📊 Feature Importance Ranking:")
    print(df_imp.to_string(index=False))

# =========================
# Helper functions
# =========================
def validate_surface(s: str) -> str | None:
    s = s.strip().capitalize()
    return s if s in VALID_SURFACES else None

def get_latest(name: str, data: pd.DataFrame):
    """
    Get the most recent match for a player and return their rank and age.

    Returns:
        (rank, age) if found, else (None, None)
    """
    player_mask = (data['p1_name'] == name) | (data['p2_name'] == name)
    player_matches = data.loc[player_mask]

    if player_matches.empty:
        return None, None

    latest_match = player_matches.sort_values('tourney_date', ascending=False).iloc[0]

    if latest_match['p1_name'] == name:
        return latest_match['p1_rank'], latest_match['p1_age']
    else:
        return latest_match['p2_rank'], latest_match['p2_age']

def get_surf_record(player: str, surf: str, surf_hist: dict) -> tuple[int,int]:
    """
    Get a player's historical surface record (wins, total matches).
    Returns (wins, total) or (0,0) if no history.
    """
    if player in surf_hist and surf in surf_hist[player]:
        return surf_hist[player][surf]
    return 0, 0

def compute_h2h(p1: str, p2: str, h2h_hist: dict) -> tuple[int,str]:
    """
    Returns (diff, message) for head-to-head history between p1 and p2.
    """
    key = tuple(sorted([p1, p2]))
    if key not in h2h_hist:
        return 0, "No prior matches"
    
    w1, w2 = h2h_hist[key]
    p1_wins = w1 if p1 == key[0] else w2
    p2_wins = w2 if p1 == key[0] else w1
    diff = p1_wins - p2_wins

    if p1_wins == p2_wins:
        msg = f"Tied {p1_wins}-{p2_wins}"
    else:
        leader = p1 if p1_wins > p2_wins else p2
        msg = f"{leader} leads {max(p1_wins, p2_wins)}-{min(p1_wins, p2_wins)}"
    
    return diff, msg

def display_matchup(
    p1: str, p2: str, surf: str, p1_rank: float, p2_rank: float, 
    p1_age: float, p2_age: float, p1_pct: float, p2_pct: float,
    p1_w: int, p1_t: int, p2_w: int, p2_t: int, h2h_msg: str, prob: float
) -> None:
    print(f"\n📊 MATCHUP STATS: {surf} Court")
    print(f"{'':<20} {p1:<20} {p2:<20}")
    print(f"{'Rank':<20} #{int(p1_rank):<19} #{int(p2_rank):<19}")
    print(f"{'Age':<20} {p1_age:.1f}y{'':<18} {p2_age:.1f}y")
    print(f"{'Surface Rec':<20} {p1_pct:.0%} ({p1_w}-{p1_t-p1_w}){'':<12} {p2_pct:.0%} ({p2_w}-{p2_t-p2_w})")
    print(f"{'Head-to-Head':<20} {h2h_msg}")
    print("-" * 60)
    if prob > DEFAULT_WIN_PCT:
        print(f"🏆 WINNER PREDICTION: {p1} ({prob:.1%} confidence)")
    else:
        print(f"🏆 WINNER PREDICTION: {p2} ({1-prob:.1%} confidence)")
    print("-" * 60 + "\n")

def build_feature_row(
    p1_rank: float,
    p2_rank: float,
    p1_age: float,
    p2_age: float,
    p1_pct: float,
    p2_pct: float,
    h2h_diff: int
) -> pd.DataFrame:
    """
    Build a single-row DataFrame matching MODEL_FEATURES order.
    """
    return pd.DataFrame(
        [[p1_rank, p2_rank, p1_age, p2_age, p1_pct, p2_pct, h2h_diff]],
        columns=MODEL_FEATURES
    )

# ==========================================
# 5. INTERACTIVE PREDICTION LOOP
# ==========================================
def interactive_prediction_loop(model, data, surf_hist, h2h_hist):
    """
    Runs a REPL-style loop for predicting tennis matches.
    Prompts for two player names and a surface, then predicts
    the winner using the provided model and historical stats.
    """
    print("\n" + "="*40)
    print(" 🎾  TENNIS MATCH PREDICTOR  🎾")
    print("="*40)
    print("Type 'exit' to quit. Use Ctrl+C to stop safely.\n")

    while True:
        try:
            p1 = input("Enter Player 1 (e.g. Carlos Alcaraz): ").strip()
            if p1.lower() == 'exit': break
            p2 = input("Enter Player 2 (e.g. Jannik Sinner): ").strip()
            if p2.lower() == 'exit': break

            surf_input = input("Enter Surface (Hard, Clay, Grass): ")
            surf = validate_surface(surf_input)

            if surf is None:
                print("❌ Invalid surface. Choose Hard, Clay, or Grass.\n")
                continue

            p1_stats = get_latest(p1, data)
            p2_stats = get_latest(p2, data)

            if not p1_stats[0] or not p2_stats[0]:
                print(f"❌ Error: Player not found in database.\n")
                continue
                
            p1_rank, p1_age = p1_stats
            p2_rank, p2_age = p2_stats
            
            p1_w, p1_t = get_surf_record(p1, surf, surf_hist)
            p2_w, p2_t = get_surf_record(p2, surf, surf_hist)

            p1_pct = p1_w/p1_t if p1_t > 0 else DEFAULT_WIN_PCT
            p2_pct = p2_w/p2_t if p2_t > 0 else DEFAULT_WIN_PCT

            diff, h2h_msg = compute_h2h(p1, p2, h2h_hist)

            # Predict
            input_data = build_feature_row(
                p1_rank, p2_rank,
                p1_age, p2_age,
                p1_pct, p2_pct,
                diff
            )
            prob = model.predict_proba(input_data)[0][1] # Probability P1 wins

            display_matchup(p1, p2, surf, p1_rank, p2_rank, p1_age, p2_age, p1_pct, p2_pct, p1_w, p1_t, p2_w, p2_t, h2h_msg, prob)

        except KeyboardInterrupt:
            print("\n\n👋 Exiting. Thank you!")
            sys.exit()
        except Exception as e:
            print(f"An error occurred: {e}")

def load_cached_data(
    path: Path,
    start_year: int,
    end_year: int
) -> pd.DataFrame | None:
    """
    Load cached ATP data if it exists and covers the required year range.
    Returns None if cache is missing or outdated.
    """
    if not path.exists():
        return None

    print(f"📂 Loading cached data from {path}...")
    df = pd.read_csv(path)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format="%Y%m%d", errors="coerce")

    cached_min = df['year'].min()
    cached_max = df['year'].max()

    if cached_min > start_year or cached_max < end_year:
        print(f"⚠️  Cache outdated (Have {cached_min}-{cached_max}, need {start_year}-{end_year})")
        return None

    return df

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    # Define paths
    DATA_PATH = Path("atp_tennis_data.csv")
    MODEL_PATH = Path("tennis_model.pkl")

    # 1. Load Data (Smart Caching)
    data = load_cached_data(DATA_PATH, START_YEAR, END_YEAR)

    if data is None:
        data = load_atp_data(START_YEAR, END_YEAR)
        data.to_csv(DATA_PATH, index=False)
        print(f"💾 New data saved to {DATA_PATH}")

    # 2. Process & Engineer Features
    # We always re-run this to ensure history dicts are fresh for the interactive loop
    processed_data = preprocess_data(data)
    final_df, surf_history, h2h_history = add_features(processed_data)
    
    # 3. Model Training (with Caching)
    if MODEL_PATH.exists():
        print(f"📂 Loading trained model from {MODEL_PATH}...")
        rf_model = joblib.load(MODEL_PATH)
    else:
        rf_model = train_and_evaluate(final_df)
        joblib.dump(rf_model, MODEL_PATH)
        print(f"💾 Model saved to {MODEL_PATH}")

    # 4. Interactive Mode
    plot_feature_importance(rf_model)
    interactive_prediction_loop(rf_model, final_df, surf_history, h2h_history)

if __name__ == "__main__":
    main()