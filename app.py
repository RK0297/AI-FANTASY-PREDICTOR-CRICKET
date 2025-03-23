import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io
st.title("Fantasy Cricket Predictor")
from data_processor import load_data, preprocess_data
from model import train_random_forest_model, predict_player_performances, DummyModel
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
This application helps you select an optimal Dream11 fantasy cricket team based on
player stats, playing status, and lineup position. Upload your player data file to get started.
""")

# Initialize file upload state
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

# File uploader for player data
uploaded_file = st.file_uploader("Upload player data (CSV)", type=["csv"])

if uploaded_file is not None:
    st.session_state.file_uploaded = True
    
    # Load and preprocess data
    try:
        df = load_data(uploaded_file)
        
        # Preview data
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Process data
        processed_data = preprocess_data(df)
        
        # Set default values for player boosts
        playing_boost = 100
        substitute_boost = 10
        
        # Budget constraint
        budget = st.sidebar.slider("Budget (credits)", 90.0, 100.0, 100.0, 0.5)
        
        # Create a button with a more appropriate name
        if 'IsPlaying' in df.columns:
            generate_button = st.sidebar.button("Generate Optimal Team Based on Playing Status")
        else:
            # If we're still using the old format with RandomForest model
            # Add model parameters
            st.sidebar.header("Model Parameters")
            n_estimators = st.sidebar.slider("Number of trees", 50, 500, 100, 50)
            max_depth = st.sidebar.slider("Max depth", 3, 20, 10, 1)
            generate_button = st.sidebar.button("Generate Optimal Team")
        
        if generate_button:
            with st.spinner("Analyzing data and generating optimal team..."):
                # Train model (or use dummy model for playing-status based prediction)
                if 'IsPlaying' in df.columns:
                    # For playing status-based prediction, we've already calculated predicted points
                    # in the preprocessing step, so we use a dummy model
                    model = DummyModel()
                    # Update priority scores based on slider settings
                    processed_data['priority_score'] = processed_data['predicted_points'].copy()
                    processed_data.loc[processed_data['IsPlaying'] == 'PLAYING', 'priority_score'] += playing_boost
                    processed_data.loc[processed_data['IsPlaying'] == 'X_FACTOR_SUBSTITUTE', 'priority_score'] += substitute_boost
                else:
                    # Traditional RandomForest model for historical stats
                    model = train_random_forest_model(
                        processed_data,
                        n_estimators=n_estimators,
                        max_depth=max_depth
                    )
                
                # Make predictions (or use pre-calculated predictions for playing-status model)
                players_with_predictions = predict_player_performances(model, processed_data)
                
                # Select optimal team
                optimal_team = select_optimal_team(players_with_predictions, budget=budget)
                
                # Validate team
                is_valid, message = validate_team(optimal_team)
                
                # Only show feature importance for traditional model (not dummy model)
                if not isinstance(model, DummyModel):
                    st.header("Model Analysis")
                    
                    # Feature importance plot
                    feature_importances = model.feature_importances_
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
                    # Show playing status if available
                    if 'IsPlaying' in optimal_team.columns:
                        display_cols = ['Player', 'Team', 'Role', 'Credits', 'IsPlaying', 'predicted_points', 'C/VC']
                    else:
                        display_cols = ['Player', 'Team', 'Role', 'Credits', 'predicted_points', 'C/VC']
                    
                    display_team(optimal_team, "Optimal Dream11 Team", display_cols)
                    
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
                    st.warning("Try adjusting the budget or selection parameters.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your input file format and try again.")

# Empty sidebar end
st.sidebar.markdown("")
