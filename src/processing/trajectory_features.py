import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import click_features as cf
from ..utils import utils

MICRO_PAUSE = 100 #ms

def segment_mouse_actions(df, dist_threshold=10, pause_threshold=1000):
    """Split mouse trajectory into movement segments based on action, using vector operations.
    Point and click [pc], Drag and drop [dd], Mouse movement[mm]"""

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

        if duration_ms >= min_duration_ms and dist > 0:
            filtered.append(seg)

    
    return filtered

def apply_sliding_window(df_user, user_id, window_size=5, step=1):
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


def compute_trajectory_features(df, user_id):
    """Compute trajectory-level features""" 

    df = df.copy()   
    #utils.plot_feature_histogram(df, 'dt')  # DEBUG - plot velocity histogram
    trajectories = segment_mouse_actions(df)
    #utils.write_all_segments_to_csv(trajectories, "segments.csv")  # DEBUG - write all segments to csv
    trajectories = clear_short_segments(trajectories)
    #utils.write_all_segments_to_csv(trajectories, "cleared_segments.csv") 
    all_features = []
    segment_id = 0


    for df_trajectory in trajectories:
        df_trajectory = df_trajectory.copy()

        # Time total
        time_duration = df_trajectory['client timestamp'].iloc[-1] - df_trajectory['client timestamp'].iloc[0]
        #print(f"Segment {segment_id}: duration {time_duration} ms, rows: {len(df_trajectory)}")

        # Path length 
        path_length = df_trajectory['dist'].sum()
        
        # Net displacement 
        start_x, start_y = df_trajectory.iloc[0][['x', 'y']]
        end_x, end_y = df_trajectory.iloc[-1][['x', 'y']]
        net_displacement = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        
        # Straightness ratio
        straightness_ratio = net_displacement / path_length if path_length > 0 else 0
        
        # sign changes in dx / dy 
        dx_signs = np.sign(df_trajectory['dx'])
        dy_signs = np.sign(df_trajectory['dy'])
        dx_changes = (dx_signs.shift(1) * dx_signs < 0).sum()
        dy_changes = (dy_signs.shift(1) * dy_signs < 0).sum()
        num_direction_changes = dx_changes + dy_changes

        #trajectory of center of mass
        time = df_trajectory['client timestamp'] - df_trajectory['client timestamp'].iloc[0]
        duration = time.iloc[-1]
        weighted_time = (time * df_trajectory['dist']).sum()
        tcm = weighted_time / path_length if path_length > 0 else 0
        tcm_norm = tcm / duration if duration > 0 else 0

        #scattering coefficient - how much the trajectory is scattered around its center of mass
        dist_i = df_trajectory['dist'].iloc[:-1]
        time_i1 = time.iloc[1:]

        sc = ((time_i1 - tcm) ** 2 * dist_i).sum() / path_length if path_length > 0 else 0
        sc_norm = sc / (duration ** 2) if duration > 0 else 0 

        #num pauses
        pauses = (df_trajectory['dt'] > MICRO_PAUSE).sum()



        all_features.append({
            'user_id': user_id,
            'segment_id': segment_id,
            'num_events': len(df_trajectory),

            'median_vel': df_trajectory['vel'].median(),
            'std_vel': df_trajectory['vel'].std(),
            'p10_vel': df_trajectory['vel'].quantile(0.1),
            'p90_vel': df_trajectory['vel'].quantile(0.9),
            'skew_vel': df_trajectory['vel'].skew(),
            'kurtosis_vel': df_trajectory['vel'].kurtosis(),

            'std_acc': df_trajectory['acc'].std(),
            'mean_abs_acc': df_trajectory['acc'].abs().mean(),
            'skew_acc': df_trajectory['acc'].skew(),
            'kurtosis_acc': df_trajectory['acc'].kurtosis(),

            'mean_jerk': df_trajectory['jerk'].mean(),
            'std_jerk': df_trajectory['jerk'].std(),
            'skew_jerk': df_trajectory['jerk'].skew(),
            'kurtosis_jerk': df_trajectory['jerk'].kurtosis(),

            'std_dt' : df_trajectory['dt'].std(),
            'std_angle': df_trajectory['angle'].std(),
            'mean_angle_change': df_trajectory['angle_change'].abs().mean(),
            'straightness_ratio': straightness_ratio,
            'num_direction_changes': num_direction_changes,
            'tcm_norm': tcm_norm,
            'scattering_coefficient_norm': sc_norm,
            'sum_angle_change': df_trajectory['angle_change'].abs().sum(),

            'num_pauses': pauses,  
            'time_duration': time_duration,
            'click_duration': cf.compute_click_duration(df_trajectory),

            'is_pc': int(df_trajectory['segment_type'].iloc[0] == 'pc') if not df_trajectory.empty else 0,
            'is_dd': int(df_trajectory['segment_type'].iloc[0] == 'dd') if not df_trajectory.empty else 0,
        })

        segment_id += 1


    
    # convert to DataFrame
    features_df = pd.DataFrame(all_features)
    features_df = apply_sliding_window(features_df, user_id, window_size=12, step=1)

    #features_df.to_csv("splitted_features.csv", index=False)

    return features_df

    



 