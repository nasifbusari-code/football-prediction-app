def predict_final_points(over_under, points_per_quarter, current_quarter_points, current_quarter, minutes_remaining):
    # Constants for standard basketball game (NBA rules)
    total_game_minutes = 48  # 4 quarters x 12 minutes
    minutes_per_quarter = 12
    
    # Quarter-specific scoring adjustments (based on typical NBA trends)
    quarter_weights = {1: 1.05, 2: 1.0, 3: 1.0, 4: 0.95}  # Q4 slightly lower scoring
    
    # Validate inputs
    if current_quarter not in [1, 2, 3, 4]:
        return "Invalid quarter. Please enter 1, 2, 3, or 4."
    if minutes_remaining < 0 or minutes_remaining > minutes_per_quarter:
        return f"Invalid minutes remaining. Must be between 0 and {minutes_per_quarter}."
    if current_quarter_points < 0:
        return "Current quarter points cannot be negative."
    
    # Calculate points so far (completed quarters + current quarter)
    points_so_far = sum(points_per_quarter) + current_quarter_points
    
    # Calculate time elapsed
    completed_quarters = current_quarter - 1
    minutes_elapsed_in_current_quarter = minutes_per_quarter - minutes_remaining
    total_minutes_elapsed = (completed_quarters * minutes_per_quarter) + minutes_elapsed_in_current_quarter
    
    # Calculate total minutes remaining
    total_minutes_remaining = total_game_minutes - total_minutes_elapsed
    
    # Observed scoring rate (overall)
    observed_rate = points_so_far / total_minutes_elapsed if total_minutes_elapsed > 0 else 0
    
    # Current quarter scoring rate (if applicable)
    current_quarter_rate = (
        current_quarter_points / minutes_elapsed_in_current_quarter
        if minutes_elapsed_in_current_quarter > 0 else observed_rate
    )
    
    # Expected scoring rate from over/under
    expected_rate = over_under / total_game_minutes
    
    # Dynamic blending based on game progress
    game_progress = total_minutes_elapsed / total_game_minutes
    observed_weight = 0.3 + 0.5 * game_progress  # 30% in Q1, up to 80% in Q4
    current_quarter_weight = 0.2 if current_quarter_rate > 0 else 0  # 20% weight to current quarter
    expected_weight = 1.0 - observed_weight - current_quarter_weight
    
    # Blend rates
    blended_rate = (
        observed_weight * observed_rate +
        current_quarter_weight * current_quarter_rate +
        expected_weight * expected_rate
    )
    
    # Adjust for remaining quarters
    remaining_points = 0
    minutes_left_in_current_quarter = minutes_remaining
    current_quarter_adjustment = quarter_weights.get(current_quarter, 1.0)
    remaining_points += blended_rate * minutes_left_in_current_quarter * current_quarter_adjustment
    
    # Add points for remaining full quarters
    for q in range(current_quarter + 1, 5):
        remaining_points += blended_rate * minutes_per_quarter * quarter_weights.get(q, 1.0)
    
    # Total predicted points
    total_predicted_points = points_so_far + remaining_points
    
    # Estimate prediction range (±10% for simplicity, based on typical game variance)
    range_margin = total_predicted_points * 0.1
    lower_bound = round(total_predicted_points - range_margin)
    upper_bound = round(total_predicted_points + range_margin)
    
    return {
        "predicted_points": round(total_predicted_points),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "over_under": over_under,
        "favor": "OVER" if total_predicted_points > over_under else "UNDER"
    }

def get_user_inputs():
    # Get pre-match over/under odds
    while True:
        try:
            over_under = float(input("Enter the pre-match over/under odds point (e.g., 210.5): "))
            if over_under <= 0:
                print("Over/under must be positive.")
                continue
            break
        except ValueError:
            print("Please enter a valid number for over/under.")
    
    # Get current quarter
    while True:
        try:
            current_quarter = int(input("Enter the current quarter (1, 2, 3, or 4): "))
            if current_quarter not in [1, 2, 3, 4]:
                print("Invalid quarter. Please enter 1, 2, 3, or 4.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer for the quarter.")
    
    # Get points for completed quarters
    points_per_quarter = []
    completed_quarters = current_quarter - 1
    for i in range(completed_quarters):
        while True:
            try:
                points = int(input(f"Enter total points scored in Quarter {i+1} (e.g., 57): "))
                if points < 0:
                    print("Points cannot be negative.")
                    continue
                points_per_quarter.append(points)
                break
            except ValueError:
                print("Please enter a valid integer for points.")
    
    # Get points scored so far in current quarter
    while True:
        try:
            current_quarter_points = int(input(f"Enter total points scored so far in Quarter {current_quarter} (e.g., 10): "))
            if current_quarter_points < 0:
                print("Points cannot be negative.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer for points.")
    
    # Get minutes remaining in current quarter
    while True:
        try:
            minutes_remaining = float(input(f"Enter minutes remaining in Quarter {current_quarter} (0 to 12): "))
            if minutes_remaining < 0 or minutes_remaining > 12:
                print("Minutes remaining must be between 0 and 12.")
                continue
            break
        except ValueError:
            print("Please enter a valid number for minutes remaining.")
    
    return over_under, points_per_quarter, current_quarter_points, current_quarter, minutes_remaining

# Main program
def main():
    print("Basketball Total Points Predictor")
    over_under, points_per_quarter, current_quarter_points, current_quarter, minutes_remaining = get_user_inputs()
    result = predict_final_points(over_under, points_per_quarter, current_quarter_points, current_quarter, minutes_remaining)
    if isinstance(result, str):
        print(result)  # Error message
    else:
        print(f"Predicted total points: {result['predicted_points']}")
        print(f"Prediction range: {result['lower_bound']} - {result['upper_bound']}")
        print(f"Compared to over/under ({result['over_under']}): Favoring {result['favor']}")

if __name__ == "__main__":
    main()
