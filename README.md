IMDb Movie Rating Analysis and Prediction
This project involves analyzing and predicting IMDb movie ratings using machine learning techniques. The dataset comprises Indian movies and their various attributes,
such as genres, directors, actors, votes, and release year. The goal was to build a predictive model for IMDb ratings and derive meaningful insights through visualizations.

Project Overview
This project was part of my CodSoft Data Science Internship, focusing on applying machine learning to solve real-world problems. Key tasks included:

Cleaning and preprocessing data.
Engineering features to enhance model performance.
Building and optimizing a machine learning model using XGBoost.
Conducting exploratory data analysis (EDA) with visualizations.
Evaluating model performance using metrics like RMSE and R².

Dataset
The dataset used is IMDb Movies India.csv, containing features such as:
Name: Movie title.
Year: Year of release.
Genre: Movie genres (e.g., Drama, Action, Comedy).
Director: Director of the movie.
Actors: Main actors.
Rating: IMDb rating of the movie.
Votes: Number of votes received.

Project Workflow
1. Data Cleaning:
Handled missing values in key columns (Rating, Genre, Director, etc.).
Converted Year, Duration, and Votes into appropriate formats.
Log-transformed Votes to reduce skewness.
2. Feature Engineering:
One-hot encoded genres.
Used LabelEncoder for categorical columns like Director and Actors.
Created interaction features (e.g., Director_Genre).
Categorized movies into Classic, Modern, and Contemporary based on the release year.
3. Model Building:
Used XGBoost Regressor for rating prediction.
Optimized hyperparameters using GridSearchCV.
Evaluated performance with metrics like RMSE, R², and cross-validation scores.
4. Visualizations:
Distribution of movie ratings.
Top genres, directors, and actors by ratings.
Yearly trends in movie ratings.
Interaction between features and ratings using heatmaps and scatterplots.

Results
Best Model Parameters:

json
Copy code
{'n_estimators': 1000, 'max_depth': 10, 'learning_rate': 0.1, 'subsample': 1.0, 'colsample_bytree': 1.0}

Evaluation Metrics:
Mean Squared Error (MSE): 1.28      
Root Mean Squared Error (RMSE): 1.13
R-squared (R²): 0.31

Visualizations
Key insights derived from visualizations:
Distribution of ratings shows a tendency toward mid-to-high ratings.
Drama and Action genres dominate the dataset in terms of count.
Directors and actors significantly influence ratings, with some consistently scoring high.
Votes strongly correlate with ratings, as more popular movies tend to have higher ratings.

Challenges
Cleaning the dataset required significant effort due to missing and inconsistent values.
Feature engineering was critical to improving model performance but required domain understanding.
Improving model accuracy was challenging.

How to Use
Follow these simple steps to get started with this project:
Clone the Repository
Open your terminal and run:

git clone https://github.com/Mandlagoniwe6/CodsoftDATask02.git  
cd CodsoftDATask02  

Install Required Libraries
Ensure you have Python installed, then install the dependencies using:
pip install -r requirements.txt  

Prepare the Dataset
Place your dataset file (e.g., movie_data.csv) in your directory.

Run the Code
Visualize Results
After running the script:
Check the console for model training details.
View the generated plots and charts directly.

Explore the Visualizations
The script generates insightful visualizations, including:
Feature importance rankings
Distribution of movie ratings
Genre-specific insights
Actual vs. Predicted ratings comparisons

Acknowledgments
This project was completed as part of the CodSoft Data Science Internship. Special thanks to CodSoft for the opportunity and guidance.
