import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io
from data_processor import load_data, preprocess_data
from model import train_random_forest_model, predict_player_performances
from team_selector import select_optimal_team
from utils import validate_team, display_team, generate_csv_download

# Set page config
st.set_page_config(
    page_title="Dream11 Fantasy Cricket Team Optimizer",
    page_icon="🏏",
    layout="wide"
)

# App title and description
st.title("Dream11 Fantasy Cricket Team Optimizer")
st.markdown("""
This application helps you select an optimal Dream11 fantasy cricket team using 
machine learning. Upload your player data file to get started.
""")

# File uploader for player data
uploaded_file = st.file_uploader("Upload player data (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load and preprocess data
    try:
        df = load_data(uploaded_file)
        
        # Preview data
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Process data
        processed_data = preprocess_data(df)
        
        # Model training parameters
        st.sidebar.header("Model Parameters")
        n_estimators = st.sidebar.slider("Number of trees", 50, 500, 100, 50)
        max_depth = st.sidebar.slider("Max depth", 3, 20, 10, 1)
        
        # Team selection parameters
        st.sidebar.header("Team Selection Parameters")
        
        # Match selection (for future implementation with schedule data)
        # teams = df['Team'].unique().tolist()
        # team1 = st.sidebar.selectbox("Select Team 1", teams)
        # team2 = st.sidebar.selectbox("Select Team 2", teams, index=1 if len(teams) > 1 else 0)
        
        # Budget constraint
        budget = st.sidebar.slider("Budget (credits)", 90, 100, 100, 0.5)
        
        if st.sidebar.button("Generate Optimal Team"):
            with st.spinner("Training model and generating optimal team..."):
                # Train model
                model = train_random_forest_model(
                    processed_data,
                    n_estimators=n_estimators,
                    max_depth=max_depth
                )
                
                # Get feature importance
                feature_importances = model.feature_importances_
                
                # Make predictions
                players_with_predictions = predict_player_performances(model, processed_data)
                
                # Select optimal team
                optimal_team = select_optimal_team(players_with_predictions, budget=budget)
                
                # Validate team
                is_valid, message = validate_team(optimal_team)
                
                st.header("Model Analysis")
                
                # Feature importance plot
                st.subheader("Feature Importance")
                fig, ax = plt.subplots(figsize=(10, 6))
                features = processed_data.drop(['Player', 'Team', 'Role', 'Credits', 'predicted_points'], axis=1).columns
                indices = np.argsort(feature_importances)[-10:]  # Top 10 features
                
                ax.barh(range(len(indices)), feature_importances[indices])
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([features[i] for i in indices])
                ax.set_xlabel('Feature Importance')
                ax.set_title('Top 10 Features by Importance')
                st.pyplot(fig)
                
                # Display team selection results
                st.header("Team Selection Results")
                
                if is_valid:
                    display_team(optimal_team, "Optimal Dream11 Team")
                    
                    # Generate CSV for download
                    csv_data = generate_csv_download(optimal_team)
                    
                    # Download button for CSV
                    st.download_button(
                        label="Download Team CSV",
                        data=csv_data,
                        file_name="dream11_team.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"Could not generate a valid team: {message}")
                    st.warning("Try adjusting the budget or model parameters.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your input file format and try again.")

# Add information section
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("""
This app uses RandomForest models to predict player performances
and optimize Dream11 fantasy cricket team selection.

**Rules Applied:**
- 11 Players per team
- At least 1 player from each team
- At least 1 player from each role (WK, BAT, AR, BOWL)
- Budget constraint (default: 100 credits)
""")
