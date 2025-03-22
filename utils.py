import streamlit as st
import pandas as pd
import io

def validate_team(team_df):
    """
    Validate if the team meets all Dream11 requirements
    
    Args:
        team_df: DataFrame with selected team
        
    Returns:
        tuple: (is_valid, message)
    """
    if team_df is None:
        return False, "No team selected"
    
    # Check if we have 11 players
    if len(team_df) != 11:
        return False, f"Team must have exactly 11 players, but has {len(team_df)}"
    
    # Check if we have at least one player from each role
    role_patterns = {
        'Wicketkeeper': 'Wicketkeeper',
        'Batter': 'Batter',
        'All-rounder': 'All-rounder',
        'Bowler': 'Bowler'
    }
    
    for role, pattern in role_patterns.items():
        if sum(team_df['Role'].str.contains(pattern)) == 0:
            return False, f"Team must have at least one {role}"
    
    # Check if we have at least one player from each team
    team_counts = team_df['Team'].value_counts()
    if len(team_counts) < 2:  # Assuming there should be at least 2 teams
        return False, "Team must have players from at least 2 different teams"
    
    # Check captain and vice-captain assignment
    if 'C/VC' in team_df.columns:
        if sum(team_df['C/VC'] == 'C') != 1:
            return False, "Team must have exactly one captain"
        if sum(team_df['C/VC'] == 'VC') != 1:
            return False, "Team must have exactly one vice-captain"
    
    # Calculate total credits
    total_credits = team_df['Credits'].sum()
    if total_credits > 100:
        return False, f"Team exceeds credit limit (100). Current: {total_credits:.1f}"
    
    return True, "Team is valid"

def display_team(team_df, title, display_cols=None):
    """
    Display the selected team in the Streamlit app
    
    Args:
        team_df: DataFrame with selected team
        title: Title to display above the team
        display_cols: Optional list of columns to display
    """
    st.subheader(title)
    
    # Calculate team stats
    total_credits = team_df['Credits'].sum()
    predicted_points = team_df['predicted_points'].sum()
    
    # Display team stats
    st.markdown(f"**Total Credits:** {total_credits:.1f}/100")
    st.markdown(f"**Predicted Points:** {predicted_points:.1f}")
    
    # Display role counts
    role_counts = {
        'WK': sum(team_df['Role'].str.contains('Wicketkeeper')),
        'BAT': sum(team_df['Role'].str.contains('Batter') & ~team_df['Role'].str.contains('Wicketkeeper')),
        'AR': sum(team_df['Role'].str.contains('All-rounder')),
        'BOWL': sum(team_df['Role'].str.contains('Bowler'))
    }
    
    st.markdown(f"**Team Composition:** WK: {role_counts['WK']} | BAT: {role_counts['BAT']} | AR: {role_counts['AR']} | BOWL: {role_counts['BOWL']}")
    
    # If playing status is available, show count of playing players
    if 'IsPlaying' in team_df.columns:
        playing_count = sum(team_df['IsPlaying'] == 'PLAYING')
        substitute_count = sum(team_df['IsPlaying'] == 'X_FACTOR_SUBSTITUTE')
        st.markdown(f"**Playing Status:** PLAYING: {playing_count} | SUBSTITUTE: {substitute_count} | NOT PLAYING: {11-playing_count-substitute_count}")
    
    # Display team table
    if display_cols is None:
        display_cols = ['Player', 'Team', 'Role', 'Credits', 'predicted_points', 'C/VC']
    
    # Ensure we only show columns that exist in the dataframe
    display_cols = [col for col in display_cols if col in team_df.columns]
    
    # Create a styled dataframe
    st.dataframe(
        team_df[display_cols]
    )
    
    # Display captain and vice-captain
    if 'C/VC' in team_df.columns:
        captain = team_df[team_df['C/VC'] == 'C']['Player'].values[0] if sum(team_df['C/VC'] == 'C') > 0 else "None"
        vice_captain = team_df[team_df['C/VC'] == 'VC']['Player'].values[0] if sum(team_df['C/VC'] == 'VC') > 0 else "None"
        
        st.markdown(f"**Captain:** {captain}")
        st.markdown(f"**Vice-Captain:** {vice_captain}")

def generate_csv_download(team_df):
    """
    Generate CSV content for download
    
    Args:
        team_df: DataFrame with selected team
        
    Returns:
        String with CSV content
    """
    # Create a copy to avoid modifying the original
    output_columns = ['Player', 'Team', 'C/VC']
    
    # Add playing status if available
    if 'IsPlaying' in team_df.columns:
        output_columns.append('IsPlaying')
    
    # Select only columns that exist in the dataframe
    output_columns = [col for col in output_columns if col in team_df.columns]
    df_out = team_df[output_columns].copy()
    
    # Write to string buffer
    buffer = io.StringIO()
    df_out.to_csv(buffer, index=False)
    
    return buffer.getvalue()
