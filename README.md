# 🎾 Tennis Match Outcome Predictor

A machine learning pipeline for predicting ATP singles match outcomes using historical data and time-aware feature engineering.

This project explores whether engineered matchup features (surface performance, head-to-head history) can outperform simple ranking-based heuristics when predicting professional tennis matches.

## Project Overview
This is a binary classification system that predicts the probability that Player A defeats Player B in a given matchup based on historical stats. Unlike simple rank-based predictions, this model also takes into account player performance across different surfaces, and their head-to-head matchup against others.

* **Data Source:** Jeff Sackmann's [ATP Matches Dataset](https://github.com/JeffSackmann/tennis_atp)
* **Tech Stack:** Python, Pandas, Scikit-Learn, XGBoost, Matplotlib
* **Best Model:** Random Forest (~64% Accuracy on 2024 Test Set)

## Features Engineered
All features are computed chronologically, using only information available before each match:
1.  **Surface-Specific Win Rate:** How well does the player perform specifically on Clay/Hard/Grass?
2.  **Head-to-Head (H2H):** Relative historical dominance of one player over the other.
3.  **Dynamic Rankings:** Uses the player's rank at the time of the match, not current rank.

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas scikit-learn xgboost matplotlib seaborn
2. Run the script: python predictor.py
3. The script will:
    - Download ATP data (cached locally)
    - Engineer historical features
    - Train 4 models (Logistic Regression, Decision Tree, Random Forest, XGBoost).
    - Save performance visualisations to the same folder.
    - Launch an Interactive Predictor in the terminal where you can enter any men's singles matchup.
   