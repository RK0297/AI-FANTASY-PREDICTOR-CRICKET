import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMaximize

def optimize_team(players, budget):
    """
    Optimizes team selection using linear programming.
    
    Args:
        players (pd.DataFrame): DataFrame with columns 'Credits', 'predicted_points'
                                and optionally other columns for role constraints.
        budget (float): Total credits available.
        
    Returns:
        pd.DataFrame: Subset of players representing the selected team.
    """
    prob = LpProblem("TeamSelection", LpMaximize)
    
    # Create decision variables for each player (0 or 1)
    decision_vars = {i: LpVariable(f"player_{i}", cat="Binary") for i in players.index}
    
    # Objective: maximize total predicted points
    prob += lpSum([players.loc[i, 'predicted_points'] * decision_vars[i] for i in players.index])
    
    # Constraint: total credits must not exceed budget
    prob += lpSum([players.loc[i, 'Credits'] * decision_vars[i] for i in players.index]) <= budget
    
    # Example additional constraint: ensure at least 4 players with role 'Batsman' if available
    if 'Role' in players.columns:
        batsman_indices = players[players['Role'] == 'Batsman'].index
        prob += lpSum([decision_vars[i] for i in batsman_indices]) >= 4
    
    prob.solve()
    
    selected_indices = [i for i in players.index if decision_vars[i].varValue == 1]
    return players.loc[selected_indices]
