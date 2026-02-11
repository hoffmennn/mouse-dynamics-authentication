
def compute_session_aggregates(df):
    move_events = df[df['state'] == 'Move'].copy()
    
    if len(move_events) == 0:
        return {}
    
    features = {
        # Velocity stats
        'mean_vel': move_events['vel'].mean(),
        'median_vel': move_events['vel'].median(),
        'std_vel': move_events['vel'].std(),
        'p10_vel': move_events['vel'].quantile(0.1),
        'p90_vel': move_events['vel'].quantile(0.9),
        'max_vel': move_events['vel'].max(),
        
        # Acceleration stats
        'mean_acc': move_events['acc'].mean(),
        'std_acc': move_events['acc'].std(),
        'mean_abs_acc': move_events['acc'].abs().mean(),
        
        # Jerk stats
        'mean_jerk': move_events['jerk'].mean(),
        'std_jerk': move_events['jerk'].std(),
        
        # Timing stats
        'mean_dt': move_events['dt'].mean(),
        'std_dt': move_events['dt'].std(),
        'median_dt': move_events['dt'].median(),
        
        # Distance stats
        'mean_dist': move_events['dist'].mean(),
        'std_dist': move_events['dist'].std(),
        
        # Curvature stats
        'mean_curvature': move_events['curvature'].mean(),
        'std_curvature': move_events['curvature'].std(),
        
        # Angle variability (entropy aproximácia)
        'std_angle': move_events['angle'].std(),
        'mean_angle_change': move_events['angle_change'].abs().mean(),
        
        # Session stats
        'session_duration': df['client timestamp'].max() - df['client timestamp'].min(),
        'num_events': len(df),
        'num_moves': len(move_events),
        'prop_moves': len(move_events) / len(df) if len(df) > 0 else 0,
    }
    
    # Click rate
    num_clicks = df['is_click'].sum()
    session_time = features['session_duration'] / 1000  # ms to seconds
    features['click_rate'] = num_clicks / session_time if session_time > 0 else 0
    
    return features
