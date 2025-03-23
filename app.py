import streamlit as st
import pandas as pd
from model import train_xgboost_model, predict_player_performances
from data_processor import load_data, preprocess_data

# Title for the app
st.title("Fantasy Cricket Team Predictor")

# Load and preprocess data
data_file = "historical_match_data.csv"  # update the path if necessary
st.write("Loading historical match data...")
df = load_data(data_file)
processed_data = preprocess_data(df)

st.write("Training XGBoost model with hyperparameter tuning...")
# Train model (set use_grid_search=True to perform tuning)
model = train_xgboost_model(processed_data, use_grid_search=True)

st.write("Model training complete.")

# For predictions, ensure the input data is preprocessed correctly.
# Here, as an example, we use the same features from processed_data.
features = processed_data.drop(['Player', 'Team', 'Role', 'Credits', 'predicted_points', 'IsPlaying', 'lineupOrder'], axis=1, errors='ignore')
predictions = predict_player_performances(model, features)
processed_data['Predicted Points'] = predictions

st.write("Player Predictions:")
st.dataframe(processed_data[['Player', 'Predicted Points']])


