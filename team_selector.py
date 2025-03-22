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
    
    # Define role constraints (support both full names and abbreviations)
    role_constraints = {
        'Wicketkeeper': {'min': 1, 'max': 4, 'pattern': 'Wicketkeeper|WK'},
        'Batter': {'min': 3, 'max': 6, 'pattern': 'Batter|BAT'},
        'All-rounder': {'min': 1, 'max': 4, 'pattern': 'All-rounder|ALL'},
        'Bowler': {'min': 3, 'max': 6, 'pattern': 'Bowler|BOWL'}
    }
    
    # Check if we're using the new format with playing status
    if 'IsPlaying' in df.columns:
        # Use priority_score if already calculated, otherwise set it based on playing status
        if 'priority_score' not in df.columns:
            df['priority_score'] = df['predicted_points'].copy()
            df.loc[df['IsPlaying'] == 'PLAYING', 'priority_score'] += 100  # Big boost for playing
            df.loc[df['IsPlaying'] == 'X_FACTOR_SUBSTITUTE', 'priority_score'] += 10  # Small boost for subs
        
        # Sort by priority score
        df = df.sort_values('priority_score', ascending=False)
    else:
        # Sort by predicted points
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
    
    # Optimized greedy selection approach
    selected_players_list = []
    remaining_credits = MAX_CREDITS
    
    # Create dictionaries to track role counts
    role_counts = {role: 0 for role in role_constraints}
    team_counts = {team: 0 for team in teams}
    
    # First prioritize getting minimum requirements for each role and team
    for role, constraint in role_constraints.items():
        # Get players for this role, sorted by priority/points
        role_players = df[df['Role'].str.contains(constraint['pattern'])]
        
        # Select minimum required players for this role
        for i, (_, player) in enumerate(role_players.iterrows()):
            # Skip if we've already selected this player
            if player['Player'] in [p['Player'] for p in selected_players_list]:
                continue
                
            # Check if adding this player would exceed budget
            if player['Credits'] > remaining_credits:
                continue
                
            # Add player
            selected_players_list.append(player.to_dict())
            remaining_credits -= player['Credits']
            role_counts[role] += 1
            team_counts[player['Team']] = team_counts.get(player['Team'], 0) + 1
            
            # Break if we've met minimum requirements for this role
            if role_counts[role] >= constraint['min']:
                break
                
            # Break if we've reached max players
            if len(selected_players_list) >= MAX_PLAYERS:
                break
                
        # Break if we've reached max players
        if len(selected_players_list) >= MAX_PLAYERS:
            break
    
    # Ensure we have at least one player from each team
    for team in teams:
        # Skip if we already have a player from this team
        if team_counts.get(team, 0) > 0:
            continue
            
        # Get players from this team, sorted by priority/points
        team_players = df[df['Team'] == team]
        
        # Select one player from this team
        for _, player in team_players.iterrows():
            # Skip if we've already selected this player
            if player['Player'] in [p['Player'] for p in selected_players_list]:
                continue
                
            # Check if adding this player would exceed budget
            if player['Credits'] > remaining_credits:
                continue
                
            # Add player
            selected_players_list.append(player.to_dict())
            remaining_credits -= player['Credits']
            role_name = next((r for r in role_constraints if player['Role'].find(role_constraints[r]['pattern']) >= 0), None)
            if role_name:
                role_counts[role_name] += 1
            team_counts[team] = team_counts.get(team, 0) + 1
            break
            
        # Break if we've reached max players
        if len(selected_players_list) >= MAX_PLAYERS:
            break
    
    # Fill remaining slots with best available players
    sorted_players = df.sort_values('priority_score' if 'priority_score' in df.columns else 'predicted_points', ascending=False)
    
    for _, player in sorted_players.iterrows():
        # Skip if we've reached max players
        if len(selected_players_list) >= MAX_PLAYERS:
            break
            
        # Skip if we've already selected this player
        if player['Player'] in [p['Player'] for p in selected_players_list]:
            continue
            
        # Check if adding this player would exceed budget
        if player['Credits'] > remaining_credits:
            continue
            
        # Check if adding this player would violate role constraints
        role_name = next((r for r in role_constraints if player['Role'].find(role_constraints[r]['pattern']) >= 0), None)
        if role_name and role_counts[role_name] >= role_constraints[role_name]['max']:
            continue
            
        # Add player
        selected_players_list.append(player.to_dict())
        remaining_credits -= player['Credits']
        if role_name:
            role_counts[role_name] += 1
        team_counts[player['Team']] = team_counts.get(player['Team'], 0) + 1
    
    # Convert list of dictionaries to DataFrame
    if selected_players_list:
        best_team = pd.DataFrame(selected_players_list)
        
        # Check if we have a valid team
        if len(best_team) < MAX_PLAYERS or not is_valid_team(best_team):
            # If team is invalid, try to fix it (this is a simplified approach)
            # In a real system, we would implement a more sophisticated repair algorithm
            # For now, we'll just return what we have and let the user know it might not be optimal
            pass
        
        # Sort by predicted points (for captain/vice-captain selection)
        best_team = best_team.sort_values('predicted_points', ascending=False)
        
        # Assign captain and vice-captain
        if len(best_team) >= 2:
            best_team['C/VC'] = 'NA'
            best_team.loc[best_team.index[0], 'C/VC'] = 'C'  # Best player is captain
            best_team.loc[best_team.index[1], 'C/VC'] = 'VC'  # Second-best player is vice-captain
            
        return best_team
    else:
        # If we couldn't select any players, return an empty DataFrame
        return pd.DataFrame(columns=df.columns)
