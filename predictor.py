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

# --- CONFIGURATION ---
START_YEAR = 2015
END_YEAR = 2024

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
    Download ATP match data from Jeff Sackmann's GitHub repository
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

    # --------------------------
    # Handle missing values
    # --------------------------
    df["winner_rank"] = df["winner_rank"].fillna(2000)
    df["loser_rank"]  = df["loser_rank"].fillna(2000)

    df["winner_age"] = df["winner_age"].fillna(df["winner_age"].median())
    df["loser_age"]  = df["loser_age"].fillna(df["loser_age"].median())

    # --------------------------
    # Randomise player order
    # --------------------------
    rng = np.random.default_rng(seed=42)
    swap = rng.random(len(df)) > 0.5

    # --------------------------
    # Build Player 1 / Player 2 dataset
    # --------------------------
    new_df = pd.DataFrame({
        "tourney_date": df["tourney_date"],
        "surface": df["surface"],
        "tourney_level": df["tourney_level"],

        "p1_name": np.where(swap, df["loser_name"], df["winner_name"]),
        "p1_rank": np.where(swap, df["loser_rank"], df["winner_rank"]),
        "p1_age":  np.where(swap, df["loser_age"],  df["winner_age"]),

        "p2_name": np.where(swap, df["winner_name"], df["loser_name"]),
        "p2_rank": np.where(swap, df["winner_rank"], df["loser_rank"]),
        "p2_age":  np.where(swap, df["winner_age"],  df["loser_age"]),
    })

    # --------------------------
    # Target: did Player 1 win?
    # --------------------------
    new_df["target"] = (~swap).astype(int)

    return new_df


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def add_features(df: pd.DataFrame):
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
        if player in surface_history and surface in surface_history[player]:
            wins, total = surface_history[player][surface]
            if total > 0:
                return wins / total
            else:
                return 0.5
        return 0.5

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
        surface = row["surface"]
        p1_won = row["target"] == 1

        # --- Surface features ---
        p1_surface_pct.append(surface_win_pct(p1, surface))
        p2_surface_pct.append(surface_win_pct(p2, surface))

        # --- H2H feature ---
        pair = tuple(sorted([p1, p2]))

        if pair in h2h_history:
            wins_a, wins_b = h2h_history[pair]

            # Expanded diff calculation
            if p1 == pair[0]:
                diff = wins_a - wins_b
            else:
                diff = wins_b - wins_a
        else:
            diff = 0

        h2h_diff.append(diff)

        # --- Update histories ---
        update_surface(p1, surface, p1_won)
        update_surface(p2, surface, not p1_won)

        if pair not in h2h_history:
            h2h_history[pair] = [0, 0]

        # Expanded winner_index logic
        if p1_won and p1 == pair[0]:
            winner_index = 0
        elif not p1_won and p1 != pair[0]:
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
    print("🧠 Training models...")
    
    # Split by Year (Train: <2025, Test: 2025)
    train_mask = df['tourney_date'].dt.year < END_YEAR
    test_mask = df['tourney_date'].dt.year == END_YEAR
    
    X_train = df.loc[train_mask, MODEL_FEATURES]
    y_train = df.loc[train_mask, 'target']
    X_test  = df.loc[test_mask, MODEL_FEATURES]
    y_test  = df.loc[test_mask, 'target']
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }
    
    results = {}
    best_model = None
    best_acc = 0
    
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
    sns.barplot(x=list(results.keys()), y=list(results.values()), hue=list(results.keys()), legend=False)
    plt.title(f"Model Accuracy (Test Year: {END_YEAR})")
    plt.ylim(0.5, 0.75)
    plt.savefig("accuracy_comparison.png")
    plt.close()
    print("   [Saved accuracy_comparison.png]")
    
    return best_model

def plot_feature_importance(model):
    importances = model.feature_importances_
    df_imp = pd.DataFrame({'Feature': MODEL_FEATURES, 'Importance': importances}).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_imp, hue='Feature', legend=False, palette='viridis')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()
    
    print("\n📊 Feature Importance Ranking:")
    print(df_imp.to_string(index=False))

# ==========================================
# 5. INTERACTIVE PREDICTION LOOP
# ==========================================
def interactive_prediction_loop(model, data, surf_hist, h2h_hist):
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
            surf = input("Enter Surface (Hard, Clay, Grass): ").strip().capitalize()
            if surf.lower() == 'exit': break
            
            # --- GET DATA ---
            def get_latest(name):
                # Find most recent match for this player to get rank/age
                mask = (data['p1_name'] == name) | (data['p2_name'] == name)
                matches = data[mask].sort_values('tourney_date', ascending=False)
                if matches.empty: return None, None
                match = matches.iloc[0]
                return (match['p1_rank'], match['p1_age']) if match['p1_name'] == name else (match['p2_rank'], match['p2_age'])

            p1_stats = get_latest(p1)
            p2_stats = get_latest(p2)

            if not p1_stats[0] or not p2_stats[0]:
                print(f"❌ Error: Player not found in database.\n")
                continue
                
            p1_rank, p1_age = p1_stats
            p2_rank, p2_age = p2_stats

            # Get Surface Record (Wins/Total)
            def get_surf_record(player):
                if player in surf_hist and surf in surf_hist[player]:
                    w, t = surf_hist[player][surf]
                    return w, t
                return 0, 0
            
            p1_w, p1_t = get_surf_record(p1)
            p2_w, p2_t = get_surf_record(p2)
            p1_pct = p1_w/p1_t if p1_t > 0 else 0.5
            p2_pct = p2_w/p2_t if p2_t > 0 else 0.5

            # Get H2H
            key = tuple(sorted([p1, p2]))
            h2h_msg = "No prior matches"
            diff = 0
            if key in h2h_hist:
                w1, w2 = h2h_hist[key] # w1 is key[0] wins
                p1_wins = w1 if p1 == key[0] else w2
                p2_wins = w2 if p1 == key[0] else w1
                diff = p1_wins - p2_wins
                
                leader = p1 if p1_wins > p2_wins else p2
                if p1_wins == p2_wins: h2h_msg = f"Tied {p1_wins}-{p2_wins}"
                else: h2h_msg = f"{leader} leads {max(p1_wins, p2_wins)}-{min(p1_wins, p2_wins)}"

            # --- PREDICT ---
            input_data = pd.DataFrame([[p1_rank, p2_rank, p1_age, p2_age, p1_pct, p2_pct, diff]], columns=MODEL_FEATURES)
            prob = model.predict_proba(input_data)[0][1] # Probability P1 wins

            # --- DISPLAY ---
            print(f"\n📊 MATCHUP STATS: {surf} Court")
            print(f"{'':<20} {p1:<20} {p2:<20}")
            print(f"{'Rank':<20} #{int(p1_rank):<19} #{int(p2_rank):<19}")
            print(f"{'Age':<20} {p1_age:.1f}y{'':<18} {p2_age:.1f}y")
            print(f"{'Surface Rec':<20} {p1_pct:.0%} ({p1_w}-{p1_t-p1_w}){'':<12} {p2_pct:.0%} ({p2_w}-{p2_t-p2_w})")
            print(f"{'Head-to-Head':<20} {h2h_msg}")
            
            print("-" * 60)
            if prob > 0.5:
                print(f"🏆 WINNER PREDICTION: {p1} ({prob:.1%} confidence)")
            else:
                print(f"🏆 WINNER PREDICTION: {p2} ({1-prob:.1%} confidence)")
            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Exiting. Thank you!")
            sys.exit()
        except Exception as e:
            print(f"An error occurred: {e}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load
    data = load_atp_data(START_YEAR, END_YEAR)
    
    # 2. Process & Feature Engineering
    processed_data = preprocess_data(data)
    final_df, surf_history, h2h_history = add_features(processed_data)
    
    # 3. Train
    rf_model = train_and_evaluate(final_df)
    plot_feature_importance(rf_model)
    
    # 4. Interactive Mode
    interactive_prediction_loop(rf_model, final_df, surf_history, h2h_history)