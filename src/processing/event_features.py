import numpy as np

def compute_event_features(df):
    """Compute event-level features"""
    df = df.copy()
    
    # Time differences
    df['dt'] = df['client timestamp'].diff().fillna(0)
    df.loc[df['dt'] < 0, 'dt'] = 0  
    
    # x, y differences
    df['dx'] = df['x'].diff().fillna(0)
    df['dy'] = df['y'].diff().fillna(0)
    
    # Distance
    df['dist'] = np.sqrt(df['dx']**2 + df['dy']**2)
    
    # Velocity
    df['vel'] = np.where(df['dt'] > 0, df['dist'] / df['dt'], 0)
    df['vel'] = df['vel'].replace([np.inf, -np.inf], 0)
    
    # Acceleration - change in velocity over time
    df['acc'] = df['vel'].diff().fillna(0)
    df.loc[df['dt'] > 0, 'acc'] = df['acc'] / df['dt']
    df['acc'] = df['acc'].replace([np.inf, -np.inf], 0)
    
    # Jerk - change in acceleration over time
    df['jerk'] = df['acc'].diff().fillna(0)
    df.loc[df['dt'] > 0, 'jerk'] = df['jerk'] / df['dt']
    df['jerk'] = df['jerk'].replace([np.inf, -np.inf], 0)
    
    # Angle of movement
    df['angle'] = np.arctan2(df['dy'], df['dx'])
    
    # Angle change
    df['angle_change'] = df['angle'].diff().fillna(0)
    # normalization [-π, π]
    df['angle_change'] = np.arctan2(np.sin(df['angle_change']), np.cos(df['angle_change']))
    
    # Curvature - change in angle over distance
    df['curvature'] = np.where(df['dist'] > 0, 
                                np.abs(df['angle_change']) / df['dist'], 
                                0)
    df['curvature'] = df['curvature'].replace([np.inf, -np.inf], 0)
    
    # Click features
    df['is_click'] = df['state'].isin(['Pressed', 'Released']).astype(int)
    df['is_pressed'] = (df['state'] == 'Pressed').astype(int)
    df['is_released'] = (df['state'] == 'Released').astype(int)
    
    
    return df