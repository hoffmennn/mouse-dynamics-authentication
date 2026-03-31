import pandas as pd

MICRO_PAUSE = 100 #ms

def segment_mouse_actions(df, dist_threshold=10, pause_threshold=800):
    """Split mouse trajectory into movement segments. Point and click [pc], Drag and drop [dd], Mouse movement[mm]"""

    #utils.plot_feature_histogram(df, 'dt')
    # create mask for segments bounderies based on time gaps and button state changes
    mask = (df['client timestamp'].diff() > pause_threshold) | (df['button'] != df['button'].shift(1))
    mask = mask.fillna(False)
    
    # split into segments based on mask
    chunk_ids = mask.cumsum() 
    chunks = [group for _, group in df.groupby(chunk_ids)]

    final_segments = []
    # classify segments based on button state and distance 
    for chunk in chunks:
        chunk = chunk.copy()
        button_state = chunk.iloc[0]['button']
        
        if button_state == 'NoButton':
            current_label = 'mm'
        else:
            total_dist = chunk['dist'].sum()
            current_label = 'pc' if total_dist < dist_threshold else 'dd' #if drag is too short, consider it as click
        
        # if current segment is click and previous was movement, combine them into point and click
        if current_label == 'pc' and final_segments and final_segments[-1]['segment_type'].iloc[0] == 'mm':
            prev_mm = final_segments.pop()
            combined = pd.concat([prev_mm, chunk], ignore_index=True)
            combined['segment_type'] = 'pc' 
            final_segments.append(combined)
        else:
            chunk['segment_type'] = current_label
            final_segments.append(chunk)

    return final_segments


def clear_short_segments(segments, min_duration_ms=200):
    """Remove short trajectory segments"""
    filtered = []
    for seg in segments:
        times = (seg['client timestamp'])
        duration_ms = (times.iloc[-1] - times.iloc[0])
        dist = seg['dist'].sum()

        if duration_ms >= min_duration_ms and dist > 20:
            filtered.append(seg)

    
    return filtered


def apply_sliding_window(df_user, user_id, window_size=15, step=3):
    """
        compute rolling features
    """
    
    if len(df_user) < window_size:
        print(f"user {user_id} does not have enough segments")
        return pd.DataFrame()

    # replacing zero click durations with median duration
    if 'click_duration' in df_user.columns:
        non_zero_clicks = df_user.loc[df_user['click_duration'] > 0, 'click_duration']
        if not non_zero_clicks.empty:
            click_median = non_zero_clicks.median()
            df_user['click_duration'] = df_user['click_duration'].replace(0, click_median)


    # columns without calculation
    no_calc_cols = ['user_id', 'segment_id', 'is_pc', 'is_dd', 'csv_file']
    no_calc_cols = [c for c in no_calc_cols if c in df_user.columns]
    feature_cols = [c for c in df_user.columns if c not in no_calc_cols]

    rolled_features = df_user[feature_cols].rolling(window=window_size).agg(['mean', 'std'])

    # remove multiindex
    rolled_features.columns = [f"{col}_{func}" for col, func in rolled_features.columns]

    # add non-calculated columns
    for col in no_calc_cols:
        rolled_features[col] = df_user[col]

    # remove first NaN rows
    rolled = rolled_features.dropna().reset_index(drop=True)

    if step > 1:
        rolled = rolled.iloc[::step].reset_index(drop=True)

    cols_order = no_calc_cols + [c for c in rolled.columns if c not in no_calc_cols]
    rolled = rolled[cols_order]

    rolled = rolled.drop(columns=['segment_id', 'is_pc', 'is_dd'])
    rolled.insert(0, 'window_id', range(len(rolled)))
    rolled.to_csv("sliding_window.csv",index=False)
    return rolled

