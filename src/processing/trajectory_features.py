import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import click_features as cf
from . import segment_processing as sp
from ..utils import utils

MICRO_PAUSE = 60 #ms

def compute_net_displacement(df):
    """Vypočíta priamu vzdialenosť medzi štartom a koncom."""
    start_x, start_y = df.iloc[0][['x', 'y']]
    end_x, end_y = df.iloc[-1][['x', 'y']]
    return np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)

def compute_direction_changes(df):
    """Vypočíta počet zmien smeru v osiach x a y."""
    dx_signs = np.sign(df['dx'])
    dy_signs = np.sign(df['dy'])
    dx_changes = (dx_signs.shift(1) * dx_signs < 0).sum()
    dy_changes = (dy_signs.shift(1) * dy_signs < 0).sum()
    return dx_changes + dy_changes

def compute_tcm_metrics(df, path_length, time, duration):
    """compute normalized trajectory center of Mass (TCM) and sattering coefficient (SC)"""

    if path_length <= 0 or duration <= 0:
        return 0, 0, duration

    # Trajectory Center of Mass (TCM)
    weighted_time = (time * df['dist']).sum()
    tcm = weighted_time / path_length
    tcm_norm = tcm / duration

    # Scattering Coefficient (SC)
    dist_i = df['dist'].iloc[:-1]
    time_i1 = time.iloc[1:]
    sc = ((time_i1 - tcm) ** 2 * dist_i).sum() / path_length
    sc_norm = sc / (duration ** 2)

    return tcm_norm, sc_norm

def compute_trajectory_features(df, user_id):
    """Compute trajectory-level features""" 

    df = df.copy()   
    #utils.plot_feature_histogram(df, 'dt')  # DEBUG - plot velocity histogram
    trajectories = sp.segment_mouse_actions(df)
    #utils.write_all_segments_to_csv(trajectories, "segments.csv")  # DEBUG - write all segments to csv
    trajectories = sp.clear_short_segments(trajectories)
    #utils.write_all_segments_to_csv(trajectories, "cleared_segments.csv") 
    all_features = []


    segment_id = 0


    for df_trajectory in trajectories:

        df_trajectory = df_trajectory.copy()

        path_length = df_trajectory['dist'].sum()
        time = df['client timestamp'] - df['client timestamp'].iloc[0]
        duration = time.iloc[-1]
        net_displacement = compute_net_displacement(df_trajectory)
        num_direction_changes = compute_direction_changes(df_trajectory)
        
        tcm_norm, sc_norm = compute_tcm_metrics(df_trajectory, path_length, time, duration)

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
            'straightness_ratio': net_displacement / path_length if path_length > 0 else 0,
            'num_direction_changes': num_direction_changes,
            'tcm_norm': tcm_norm,
            'scattering_coefficient_norm': sc_norm,
            'sum_angle_change': df_trajectory['angle_change'].abs().sum(),
            'num_pauses': (df_trajectory['dt'] > MICRO_PAUSE).sum(),  
            'time_duration': duration,
            'click_duration': cf.compute_click_duration(df_trajectory),

            'is_pc': int(df_trajectory['segment_type'].iloc[0] == 'pc') if not df_trajectory.empty else 0,
            'is_dd': int(df_trajectory['segment_type'].iloc[0] == 'dd') if not df_trajectory.empty else 0,
        })

        segment_id += 1


    
    # convert to DataFrame
    features_df = pd.DataFrame(all_features)
    #features_df = sp.apply_sliding_window(features_df, user_id, window_size= 22, step=4)
    #utils.plot_feature_histogram(features_df, 'time_duration')  # DEBUG - plot velocity histogram

    #features_df.to_csv("splitted_features.csv", index=False)

    return features_df

    



 