import pandas as pd
import numpy as np
from itertools import combinations

def select_optimal_team(players_df, budget=100):
    """
    Select optimal team based on predicted performance
    
    Args:
        players_df: DataFrame with player data and predictions
        budget: Maximum budget in credits
        
    Returns:
        DataFrame containing the selected team
    """
    # Make a copy of the dataframe to avoid modifying the original
    df = players_df.copy()
    
    # Define constraints
    MAX_PLAYERS = 11
    MAX_CREDITS = budget
    MIN_PLAYERS_PER_TEAM = 1
    
    # Define role constraints
    role_constraints = {
        'Wicketkeeper': {'min': 1, 'max': 4, 'pattern': 'Wicketkeeper'},
        'Batter': {'min': 3, 'max': 6, 'pattern': 'Batter'},
        'All-rounder': {'min': 1, 'max': 4, 'pattern': 'All-rounder'},
        'Bowler': {'min': 3, 'max': 6, 'pattern': 'Bowler'}
    }
    
    # First, sort by predicted points
    df = df.sort_values('predicted_points', ascending=False)
    
    # Get unique teams
    teams = df['Team'].unique()
    
    # Function to check if a team is valid
    def is_valid_team(team_df):
        # Check if we have at least MIN_PLAYERS_PER_TEAM from each team
        team_counts = team_df['Team'].value_counts()
        for team in teams:
            if team not in team_counts or team_counts[team] < MIN_PLAYERS_PER_TEAM:
                return False
        
        # Check role constraints
        for role, constraint in role_constraints.items():
            role_count = sum(team_df['Role'].str.contains(constraint['pattern']))
            if role_count < constraint['min'] or role_count > constraint['max']:
                return False
        
        return True
    
    # Initialize best team and score
    best_team = None
    best_score = 0
    
    # First pass: greedy selection of top players by predicted points
    selected_players = pd.DataFrame(columns=df.columns)
    remaining_credits = MAX_CREDITS
    
    # Select wicketkeeper(s)
    wk_players = df[df['Role'].str.contains('Wicketkeeper')].head(4)
    for _, player in wk_players.iterrows():
        if len(selected_players) < MAX_PLAYERS and player['Credits'] <= remaining_credits:
            selected_players = pd.concat([selected_players, pd.DataFrame([player])], ignore_index=True)
            remaining_credits -= player['Credits']
            if len(selected_players[selected_players['Role'].str.contains('Wicketkeeper')]) >= role_constraints['Wicketkeeper']['min']:
                break
    
    # Select batters
    bat_players = df[df['Role'].str.contains('Batter') & ~df['Role'].str.contains('Wicketkeeper')].head(6)
    for _, player in bat_players.iterrows():
        if len(selected_players) < MAX_PLAYERS and player['Credits'] <= remaining_credits:
            if player['Player'] not in selected_players['Player'].values:
                selected_players = pd.concat([selected_players, pd.DataFrame([player])], ignore_index=True)
                remaining_credits -= player['Credits']
            if len(selected_players[selected_players['Role'].str.contains('Batter') & ~selected_players['Role'].str.contains('Wicketkeeper')]) >= role_constraints['Batter']['min']:
                break
    
    # Select all-rounders
    ar_players = df[df['Role'].str.contains('All-rounder')].head(4)
    for _, player in ar_players.iterrows():
        if len(selected_players) < MAX_PLAYERS and player['Credits'] <= remaining_credits:
            if player['Player'] not in selected_players['Player'].values:
                selected_players = pd.concat([selected_players, pd.DataFrame([player])], ignore_index=True)
                remaining_credits -= player['Credits']
            if len(selected_players[selected_players['Role'].str.contains('All-rounder')]) >= role_constraints['All-rounder']['min']:
                break
    
    # Select bowlers
    bowl_players = df[df['Role'].str.contains('Bowler')].head(6)
    for _, player in bowl_players.iterrows():
        if len(selected_players) < MAX_PLAYERS and player['Credits'] <= remaining_credits:
            if player['Player'] not in selected_players['Player'].values:
                selected_players = pd.concat([selected_players, pd.DataFrame([player])], ignore_index=True)
                remaining_credits -= player['Credits']
            if len(selected_players[selected_players['Role'].str.contains('Bowler')]) >= role_constraints['Bowler']['min']:
                break
    
    # Fill remaining slots with best available players
    remaining_players = df[~df['Player'].isin(selected_players['Player'])]
    for _, player in remaining_players.iterrows():
        if len(selected_players) < MAX_PLAYERS and player['Credits'] <= remaining_credits:
            selected_players = pd.concat([selected_players, pd.DataFrame([player])], ignore_index=True)
            remaining_credits -= player['Credits']
            if len(selected_players) == MAX_PLAYERS:
                break
    
    # Check if we have a valid team
    if len(selected_players) == MAX_PLAYERS and is_valid_team(selected_players):
        best_team = selected_players
        best_score = selected_players['predicted_points'].sum()
    
    # If no valid team found with greedy approach, try a more comprehensive search
    if best_team is None:
        # Get top players for each role
        top_wk = df[df['Role'].str.contains('Wicketkeeper')].head(4)
        top_bat = df[df['Role'].str.contains('Batter') & ~df['Role'].str.contains('Wicketkeeper')].head(8)
        top_ar = df[df['Role'].str.contains('All-rounder')].head(6)
        top_bowl = df[df['Role'].str.contains('Bowler')].head(8)
        
        # Try different combinations of players
        for wk_count in range(1, 5):
            for bat_count in range(3, 7):
                for ar_count in range(1, 5):
                    bowl_count = MAX_PLAYERS - wk_count - bat_count - ar_count
                    if bowl_count < 3 or bowl_count > 6:
                        continue
                    
                    # Try all combinations of the top players
                    for wk_combo in combinations(top_wk.iterrows(), min(wk_count, len(top_wk))):
                        wk_team = pd.DataFrame([player for _, player in wk_combo])
                        wk_credits = wk_team['Credits'].sum()
                        
                        if wk_credits > MAX_CREDITS:
                            continue
                        
                        for bat_combo in combinations(top_bat.iterrows(), min(bat_count, len(top_bat))):
                            bat_team = pd.DataFrame([player for _, player in bat_combo])
                            bat_credits = bat_team['Credits'].sum()
                            
                            if wk_credits + bat_credits > MAX_CREDITS:
                                continue
                            
                            for ar_combo in combinations(top_ar.iterrows(), min(ar_count, len(top_ar))):
                                ar_team = pd.DataFrame([player for _, player in ar_combo])
                                ar_credits = ar_team['Credits'].sum()
                                
                                if wk_credits + bat_credits + ar_credits > MAX_CREDITS:
                                    continue
                                
                                for bowl_combo in combinations(top_bowl.iterrows(), min(bowl_count, len(top_bowl))):
                                    bowl_team = pd.DataFrame([player for _, player in bowl_combo])
                                    total_credits = wk_credits + bat_credits + ar_credits + bowl_team['Credits'].sum()
                                    
                                    if total_credits > MAX_CREDITS:
                                        continue
                                    
                                    # Combine all players
                                    team = pd.concat([wk_team, bat_team, ar_team, bowl_team], ignore_index=True)
                                    
                                    # Check if valid
                                    if is_valid_team(team):
                                        team_score = team['predicted_points'].sum()
                                        if team_score > best_score:
                                            best_team = team
                                            best_score = team_score
    
    # If still no valid team, return a basic team with one of each role and team
    if best_team is None:
        selected_players = pd.DataFrame(columns=df.columns)
        remaining_credits = MAX_CREDITS
        
        # Ensure we have at least one player from each team
        for team in teams:
            team_player = df[df['Team'] == team].iloc[0] if len(df[df['Team'] == team]) > 0 else None
            if team_player is not None and team_player['Credits'] <= remaining_credits:
                selected_players = pd.concat([selected_players, pd.DataFrame([team_player])], ignore_index=True)
                remaining_credits -= team_player['Credits']
        
        # Ensure we have at least one player from each role
        for role, constraint in role_constraints.items():
            if len(selected_players[selected_players['Role'].str.contains(constraint['pattern'])]) == 0:
                role_player = df[df['Role'].str.contains(constraint['pattern'])].iloc[0] if len(df[df['Role'].str.contains(constraint['pattern'])]) > 0 else None
                if role_player is not None and role_player['Player'] not in selected_players['Player'].values and role_player['Credits'] <= remaining_credits:
                    selected_players = pd.concat([selected_players, pd.DataFrame([role_player])], ignore_index=True)
                    remaining_credits -= role_player['Credits']
        
        # Fill remaining slots with best available players
        remaining_players = df[~df['Player'].isin(selected_players['Player'])]
        for _, player in remaining_players.iterrows():
            if len(selected_players) < MAX_PLAYERS and player['Credits'] <= remaining_credits:
                selected_players = pd.concat([selected_players, pd.DataFrame([player])], ignore_index=True)
                remaining_credits -= player['Credits']
                if len(selected_players) == MAX_PLAYERS:
                    break
        
        best_team = selected_players
    
    # Sort by predicted points (for captain/vice-captain selection)
    if best_team is not None:
        best_team = best_team.sort_values('predicted_points', ascending=False)
        
        # Assign captain and vice-captain
        if len(best_team) >= 2:
            best_team['C/VC'] = 'NA'
            best_team.loc[best_team.index[0], 'C/VC'] = 'C'  # Best player is captain
            best_team.loc[best_team.index[1], 'C/VC'] = 'VC'  # Second-best player is vice-captain
    
    return best_team
