import config
import features.engineering as features
import sys
import cli.helpers as helpers

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
            surf = helpers.validate_surface(surf_input)

            if surf is None:
                print("❌ Invalid surface. Choose Hard, Clay, or Grass.\n")
                continue

            p1_stats = helpers.get_latest(p1, data)
            p2_stats = helpers.get_latest(p2, data)

            if not p1_stats[0] or not p2_stats[0]:
                print(f"❌ Error: Player not found in database.\n")
                continue
                
            p1_rank, p1_age = p1_stats
            p2_rank, p2_age = p2_stats
            
            p1_w, p1_t = helpers.get_surf_record(p1, surf, surf_hist)
            p2_w, p2_t = helpers.get_surf_record(p2, surf, surf_hist)

            p1_pct = p1_w/p1_t if p1_t > 0 else config.DEFAULT_WIN_PCT
            p2_pct = p2_w/p2_t if p2_t > 0 else config.DEFAULT_WIN_PCT

            diff, h2h_msg = helpers.compute_h2h(p1, p2, h2h_hist)

            # Predict
            input_data = helpers.build_feature_row(
                p1_rank, p2_rank,
                p1_age, p2_age,
                p1_pct, p2_pct,
                diff
            )
            prob = model.predict_proba(input_data)[0][1] # Probability P1 wins

            helpers.display_matchup(p1, p2, surf, p1_rank, p2_rank, p1_age, p2_age, p1_pct, p2_pct, p1_w, p1_t, p2_w, p2_t, h2h_msg, prob)

        except KeyboardInterrupt:
            print("\n\n👋 Exiting. Thank you!")
            sys.exit()
        except Exception as e:
            print(f"An error occurred: {e}")
