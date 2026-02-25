import numpy as np
import pandas as pd



def compute_click_features_old(df):
    
    pressed_events = df[df['state'] == 'Pressed'].copy()
    released_events = df[df['state'] == 'Released'].copy()
    
    #Press-release pairs
    click_durations = []
    for idx, press in pressed_events.iterrows():
        
        future_releases = released_events[
            (released_events['client timestamp'] > press['client timestamp']) &
            (released_events['button'] == press['button'])
        ]
        if len(future_releases) > 0:
            release = future_releases.iloc[0]
            duration = release['client timestamp'] - press['client timestamp']
            click_durations.append(duration)
    
    #Time between presses
    inter_click_times = pressed_events['client timestamp'].diff().dropna().values
    
    return {
        'num_clicks': len(pressed_events),
        'mean_click_duration': np.mean(click_durations) if click_durations else 0,
        'std_click_duration': np.std(click_durations) if click_durations else 0,
        'mean_inter_click_time': np.mean(inter_click_times) if len(inter_click_times) > 0 else 0,
        'std_inter_click_time': np.std(inter_click_times) if len(inter_click_times) > 0 else 0,
    }

def compute_click_features(df, dist_threshold=10):
    # only click-related events
    mask = df['state'].isin(['Pressed', 'Released', 'Drag'])
    events = df[mask].copy()

    events['click_id'] = (events['state'] == 'Pressed').cumsum()

    # aggregate by click_id to get click-level features
    clicks = events.groupby('click_id').agg(
        ts_start=('client timestamp', 'first'),
        ts_end=('client timestamp', 'last'),
        x_start=('x', 'first'),
        y_start=('y', 'first'),
        x_min=('x', 'min'),
        x_max=('x', 'max'),
        y_min=('y', 'min'),
        y_max=('y', 'max'),
        state_last=('state', 'last'),
        button=('button', 'first')
    ).reset_index()

    dx = clicks['x_max'] - clicks['x_min']
    dy = clicks['y_max'] - clicks['y_min']
    clicks['move_dist'] = np.sqrt(dx**2 + dy**2)

    # eliminating drag events, clicks without release
    valid_clicks = clicks[
        (clicks['state_last'] == 'Released') & 
        (clicks['move_dist'] < dist_threshold)
    ].copy()

    click_durations = valid_clicks['ts_end'] - valid_clicks['ts_start']
    inter_click_times = valid_clicks['ts_start'].diff().dropna()

    return {
        'num_clicks': len(valid_clicks),
        'mean_click_duration': click_durations.mean() if not click_durations.empty else 0,
        'std_click_duration': click_durations.std() if not click_durations.empty else 0,
        'mean_inter_click_time': inter_click_times.mean() if not inter_click_times.empty else 0,
        'std_inter_click_time': inter_click_times.std() if not inter_click_times.empty else 0,
    }


def compute_click_duration(segment):

    
    #only pressed/released events 
    mask = segment['state'].isin(['Pressed', 'Released'])
    events = segment[mask].reset_index(drop=True)
    
    if len(events) < 2:
        return 0.0
        

    is_pressed = events['state'] == 'Pressed'
    is_next_released = events['state'].shift(-1) == 'Released'
    
    pair_starts = is_pressed & is_next_released
    
    if not pair_starts.any():
        return 0.0
        
    #extract data for Pressed - Released pairs
    t1 = events.loc[pair_starts, 'client timestamp'].values
    x1 = events.loc[pair_starts, 'x'].values
    y1 = events.loc[pair_starts, 'y'].values
    
    released_indices = np.where(pair_starts)[0] + 1
    t2 = events.loc[released_indices, 'client timestamp'].values
    x2 = events.loc[released_indices, 'x'].values
    y2 = events.loc[released_indices, 'y'].values
    

    durations = t2 - t1
    distances_sq = (x2 - x1)**2 + (y2 - y1)**2
    
    valid_clicks = durations[distances_sq <= 400]
    if len(valid_clicks) == 0:
        return 0.0
        
    return np.mean(valid_clicks)