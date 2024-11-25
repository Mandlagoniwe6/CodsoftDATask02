import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

movie_data = pd.read_csv("IMDb Movies India.csv", encoding="ISO-8859-1")

# Drop rows with missing ratings
movie_data = movie_data.dropna(subset=["Rating"])

# Fill missing values for categorical columns
categorical_columns = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]
movie_data[categorical_columns] = movie_data[categorical_columns].fillna("Unknown")

# Clean numeric columns (Year, Duration, Votes)
movie_data["Year"] = movie_data["Year"].str.extract(r'(\d+)').astype(float).fillna(0).astype(int)
movie_data["Duration"] = movie_data["Duration"].str.extract(r'(\d+)').astype(float).fillna(0).astype(int)
movie_data["Votes"] = pd.to_numeric(movie_data["Votes"], errors="coerce").fillna(0).astype(int)

# Log-transform Votes to reduce skewness
movie_data["Log_Votes"] = np.log1p(movie_data["Votes"])

# One-hot encode Genre and drop the original column
genre_dummies = movie_data['Genre'].str.get_dummies(sep=', ')
movie_data = pd.concat([movie_data, genre_dummies], axis=1)

# Encode Director and Actors as numeric values using LabelEncoder
label_encoders = {}
for col in ["Director", "Actor 1", "Actor 2", "Actor 3"]:
    le = LabelEncoder()
    movie_data[col] = le.fit_transform(movie_data[col])
    label_encoders[col] = le

X = movie_data.drop(columns=["Name", "Genre", "Rating", "Votes"])
y = movie_data["Rating"]

# Feature Scaling for Numeric Features
numeric_features = ["Year", "Duration", "Log_Votes"]
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Feature Engineering: Interaction features and Year Buckets
X['Director_Genre'] = X['Director'] * X['Drama'] 
current_year = 2024
X['Year_Bucket'] = pd.cut(movie_data['Year'], bins=[1900, 1980, 2000, current_year], labels=['Classic', 'Modern', 'Contemporary'])
X = pd.get_dummies(X, columns=['Year_Bucket'], drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost model and parameter grid
xgb_model = XGBRegressor(random_state=42, objective='reg:squarederror')
param_grid = {
    'n_estimators': [1000],
    'max_depth': [10],
    'learning_rate': [0.1],
    'subsample': [1.0],
    'colsample_bytree': [1.0]
}

# Grid Search with Cross-Validation
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)
best_xgb_model = grid_search.best_estimator_

print("Best XGBoost Parameters:", grid_search.best_params_)

# Predictions and evaluation
y_pred = best_xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nEvaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Cross-Validation RMSE
cv_scores = cross_val_score(best_xgb_model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)

print("\nCross-Validation RMSE Scores:")
print(cv_rmse)
print(f"Mean CV RMSE: {cv_rmse.mean():.2f}, Std Dev: {cv_rmse.std():.2f}")

# Feature Importance Visualization
importances = best_xgb_model.feature_importances_
indices = np.argsort(importances)[-10:]  

plt.figure(figsize=(10, 6))
plt.title("Top 10 Feature Importances (XGBoost)")
plt.barh(range(len(indices)), importances[indices], align="center", color="teal")
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel("Importance")
plt.show()

# Visualization: Distribution of Movie Ratings
plt.figure(figsize=(8, 5))
sns.histplot(movie_data['Rating'], kde=True, bins=10, color="skyblue")
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Visualization: Genre Distribution
genre_counts = genre_dummies.sum().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
genre_counts.plot(kind='bar', color='teal')
plt.title('Movie Counts by Genre')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Visualization: Actual vs. Predicted Ratings
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Diagonal line
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs. Predicted Ratings")
plt.show()

# Visualization: Residual Analysis
errors = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(errors, kde=True, bins=15, color="red")
plt.title("Distribution of Prediction Errors (Residuals)")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()

# Visualization: Year vs. Rating
plt.figure(figsize=(10, 6))
sns.lineplot(data=movie_data, x="Year", y="Rating", color="blue", marker="o")
plt.title("Year vs. Average Movie Rating")
plt.xlabel("Year")
plt.ylabel("Average Rating")
plt.show()

# Visualization: Votes vs. Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(data=movie_data, x="Log_Votes", y="Rating", color="green", alpha=0.6)
plt.title("Log Votes vs. Rating")
plt.xlabel("Log of Votes")
plt.ylabel("Rating")
plt.show()

# Visualization: Top Directors by Average Rating
top_directors = (
    movie_data.groupby("Director")["Rating"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
plt.figure(figsize=(12, 6))
sns.barplot(data=top_directors, x="Rating", y="Director", palette="viridis")
plt.title("Top 10 Directors by Average Movie Rating")
plt.xlabel("Average Rating")
plt.ylabel("Director")
plt.show()

# Visualization: Duration Distribution by Rating Buckets
movie_data["Rating_Bucket"] = pd.cut(
    movie_data["Rating"], bins=[0, 5, 7, 10], labels=["Low", "Medium", "High"]
)
plt.figure(figsize=(12, 6))
sns.boxplot(data=movie_data, x="Rating_Bucket", y="Duration", palette="coolwarm")
plt.title("Duration Distribution by Rating Buckets")
plt.xlabel("Rating Bucket")
plt.ylabel("Duration (minutes)")
plt.show()

# Visualization: Average Rating by Year Buckets
average_rating_year_bucket = (
    movie_data.groupby("Year_Bucket")["Rating"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)
plt.figure(figsize=(8, 6))
sns.barplot(data=average_rating_year_bucket, x="Year_Bucket", y="Rating", palette="Blues")
plt.title("Average Movie Rating by Year Buckets")
plt.xlabel("Year Bucket")
plt.ylabel("Average Rating")
plt.show()

# Visualization: Ratings by Actor
top_actors = (
    movie_data.groupby("Actor 1")["Rating"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
plt.figure(figsize=(12, 6))
sns.barplot(data=top_actors, x="Rating", y="Actor 1", palette="mako")
plt.title("Top 10 Actors by Average Movie Rating")
plt.xlabel("Average Rating")
plt.ylabel("Actor")
plt.show()

# Visualization: Rating Distribution Across Genres
genre_ratings = movie_data[genre_dummies.columns].multiply(movie_data["Rating"], axis="index")
average_genre_ratings = genre_ratings.mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
average_genre_ratings.plot(kind="bar", color="teal")
plt.title("Average Rating by Genre")
plt.xlabel("Genre")
plt.ylabel("Average Rating")
plt.show()

# Visualization: Rating vs. Genre Combination Heatmap
genre_rating_heatmap = (
    movie_data.groupby(["Drama", "Action", "Comedy", "Romance"])["Rating"]
    .mean()
    .unstack()
)
plt.figure(figsize=(10, 8))
sns.heatmap(genre_rating_heatmap, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Rating vs. Genre Combinations")
plt.xlabel("Genre Combination")
plt.ylabel("Genre")
plt.show()

# Visualization: Votes Distribution by Rating Buckets
plt.figure(figsize=(12, 6))
sns.boxplot(data=movie_data, x="Rating_Bucket", y="Log_Votes", palette="Set3")
plt.title("Votes Distribution by Rating Buckets")
plt.xlabel("Rating Bucket")
plt.ylabel("Log of Votes")
plt.show()

# Visualization: Interaction of Director and Rating Buckets
plt.figure(figsize=(12, 6))
sns.boxplot(data=movie_data, x="Rating_Bucket", y="Director", palette="spring", showfliers=False)
plt.title("Interaction of Director and Rating Buckets")
plt.xlabel("Rating Bucket")
plt.ylabel("Director")
plt.show()
