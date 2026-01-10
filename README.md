# 🎾 Tennis Match Outcome Predictor

Machine Learning pipeline to predict ATP tennis match outcomes using historical data. This project explores whether engineered historical features can outperform simple ranking-based heuristics in predicting singles tennis match outcomes.

## 📌 Project Overview
This project builds a Binary Classification model to predict the probability that Player A wins a given match over Player B based on historical stats. Unlike simple rank-based predictions, this model engineers complex time-series features to capture player form and matchup compatibility.

* **Data Source:** Jeff Sackmann's [ATP Matches Dataset](https://github.com/JeffSackmann/tennis_atp)
* **Tech Stack:** Python, Pandas, Scikit-Learn, XGBoost, Matplotlib
* **Best Model:** Random Forest (~64% Accuracy on 2024 Test Set)

## 🔧 Features Engineered
To avoid data leakage, all features are calculated chronologically:
1.  **Surface-Specific Win Rate:** How well does the player perform specifically on Clay/Hard/Grass?
2.  **Head-to-Head (H2H):** Historical dominance of one player over the other.
3.  **Dynamic Rankings:** Uses the player's rank at the time of the match, not current rank.

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install pandas scikit-learn xgboost matplotlib seaborn
2. Run the script: python predictor.py
3. The script will:
    - Download the latest data automatically.
    - Train 4 models (Logistic Regression, Decision Tree, Random Forest, XGBoost).
    - Save performance charts to the folder.
    - Launch an Interactive Predictor in the terminal where you can enter any matchup.