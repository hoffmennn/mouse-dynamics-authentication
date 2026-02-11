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