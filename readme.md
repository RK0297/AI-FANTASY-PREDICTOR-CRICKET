# AI-Based Fantasy Dream11 Predictor

## Overview

This repository contains an AI-powered predictor model that selects the **best 11 fantasy players** for a given match based on historical data. It leverages machine learning techniques to analyze player performance, pitch conditions, and opponent statistics, ensuring optimal team selection.

## Features

- Predicts the **top 11 players** for a fantasy match.
- Ensures at least **one player from each team** is selected.
- **Automatically determines pitch type** and applies relevant constraints.
- Filters players by **date, role, and opponent statistics**.
- Outputs **a CSV file** with the best lineup along with 4 backup players.
- Uses **XGBoost** for player fantasy points prediction.

## Tech Stack

- **Frontend**: Streamlit (for UI)
- **Backend**: Pandas, NumPy, Scikit-Learn, Joblib
- **Visualization**: Matplotlib, Seaborn
- **Data Handling**: OpenPyXL (for Excel file support)
- **Performance Monitoring**: tqdm (for progress tracking)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-fantasy-predictor.git
   cd ai-fantasy-predictor
