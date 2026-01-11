import config
import pandas as pd

# =========================
# Helper functions
# =========================
def validate_surface(s: str) -> str | None:
    s = s.strip().capitalize()
    return s if s in config.VALID_SURFACES else None

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
    if prob > config.DEFAULT_WIN_PCT:
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
        columns = config.MODEL_FEATURES
    )