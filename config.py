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
