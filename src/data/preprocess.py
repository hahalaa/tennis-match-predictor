import config
import pandas as pd
import numpy as np

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
    df["winner_rank"] = df["winner_rank"].fillna(config.DEFAULT_RANK)
    df["loser_rank"]  = df["loser_rank"].fillna(config.DEFAULT_RANK)

    df["winner_age"] = df["winner_age"].fillna(df["winner_age"].median())
    df["loser_age"]  = df["loser_age"].fillna(df["loser_age"].median())

    # Randomise player order
    rng = np.random.default_rng(seed=42)
    swap_players = rng.random(len(df)) > config.DEFAULT_WIN_PCT

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