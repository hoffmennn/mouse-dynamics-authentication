import numpy as np

def compute_click_features(df):
    

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