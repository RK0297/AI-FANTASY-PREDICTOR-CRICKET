import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, make_scorer
import pandas as pd

def train_xgboost_model(data, use_grid_search=True, random_state=42):
    """
    Trains an XGBoost model using either hyperparameter tuning (GridSearchCV)
    or default parameters.
    
    Parameters:
      data (pd.DataFrame): Preprocessed data including 'predicted_points'.
      use_grid_search (bool): Set to True to perform hyperparameter tuning.
      random_state (int): Random seed for reproducibility.
    
    Returns:
      model: The trained XGBoost model.
    """
    # Prepare features and target (dropping non-numeric columns)
    features = data.drop(['Player', 'Team', 'Role', 'Credits', 'predicted_points', 'IsPlaying', 'lineupOrder'], axis=1, errors='ignore')
    target = data['predicted_points']
    
    # Split for hyperparameter tuning purposes
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=random_state)
    
    if use_grid_search:
        param_grid = {
            'n_estimators': [150, 200, 250],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9]
        }
        # Define RMSE scorer (lower is better)
        rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)
        xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state)
        grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, scoring=rmse_scorer, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        test_rmse = np.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))
        print("Best hyperparameters found:", grid_search.best_params_)
        print("Test RMSE:", test_rmse)
        return best_model
    else:
        # Train default model with preset parameters
        model = xgb.XGBRegressor(
            objective='reg:squarederror', 
            n_estimators=200, 
            max_depth=6, 
            learning_rate=0.1, 
            subsample=0.8, 
            random_state=random_state
        )
        model.fit(features, target)
        return model

def predict_player_performances(model, input_data):
    """
    Predicts player performances using the trained model.
    
    Parameters:
      model: Trained XGBoost regression model.
      input_data (pd.DataFrame): Preprocessed features (must match training features).
      
    Returns:
      array: Predicted performance values.
    """
    predictions = model.predict(input_data)
    return predictions
    
