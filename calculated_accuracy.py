from sklearn.metrics import mean_squared_error
import numpy as np

# Example data: Replace these with your actual recommendation system's output and the actual ratings
predicted_ratings = [5, 4, 3, 4, 5, 2, 3, 4, 5, 3]  # Predicted ratings for top 10 movies
actual_ratings = [5, 4, 4, 4, 5, 3, 3, 4, 5, 2]    # Actual ratings of the user

# Function to calculate accuracy, precision, and recall
def calculate_metrics(predictions, actuals, top_k=10):
    correct_recommendations = 0
    relevant_movies = 0
    total_recommendations = len(predictions)

    # Checking correct recommendations (hit ratio)
    for i in range(top_k):
        if predictions[i] == actuals[i]:  # Assuming the movies are ranked by the user
            correct_recommendations += 1

    accuracy = (correct_recommendations / total_recommendations) * 100

    # Precision at K: How many of the top K recommendations are relevant
    precision_at_k = correct_recommendations / top_k

    # Recall at K: How many of the relevant items are in the top K
    recall_at_k = correct_recommendations / len(actuals)  # assuming actuals contain relevant movies

    return accuracy, precision_at_k, recall_at_k

# Call the function with top_k set to 10
accuracy, precision_at_k, recall_at_k = calculate_metrics(predicted_ratings, actual_ratings, top_k=10)

# Output the results
print(f"Accuracy (Hit Ratio): {accuracy}%")
print(f"Precision@{10}: {precision_at_k * 100}%")
print(f"Recall@{10}: {recall_at_k * 100}%")

# rsme value
from sklearn.metrics import mean_squared_error
from math import sqrt

# Assuming you have a function that predicts movie ratings
# You can replace these variables with your actual data and predictions
# For example:
# actual_ratings = [actual user ratings]
# predicted_ratings = [predicted user ratings]

actual_ratings = [4, 5, 3, 2, 5, 4]  # Example: replace with real data
predicted_ratings = [4.2, 4.8, 3.1, 2.0, 5.0, 4.1]  # Example: replace with real predictions

# Calculate RMSE (Root Mean Squared Error)
rmse = sqrt(mean_squared_error(actual_ratings, predicted_ratings))

# Print the RMSE or use it as part of your evaluation
print(f"Root Mean Squared Error (RMSE): {rmse}")

# You can also add other metrics like Precision, Recall, or F1 Score if needed
