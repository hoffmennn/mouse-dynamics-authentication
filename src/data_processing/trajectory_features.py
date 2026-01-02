import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def split_trajectory(df):
    """Split trajectory into segments based on state changes"""
    segments = []
    current_segment = []
    current_state = None
    
    for _, row in df.iterrows():
        if row['state'] != current_state:
            if current_segment:
                segments.append(pd.DataFrame(current_segment))
                current_segment = []
            current_state = row['state']
        
        current_segment.append(row)
    
    if current_segment:
        segments.append(pd.DataFrame(current_segment))
    
    # remove short segments: duration < 200 ms 
    segments = clear_trajectory_segments(segments)

        
    return segments



def clear_trajectory_segments(segments):
    """Remove short trajectory segments"""
    filtered = []
    for seg in segments:
        
        times = (seg['client timestamp'])
        duration_ms = (times.iloc[-1] - times.iloc[0])

        #print(f"start: {times.iloc[0]}, end: {times.iloc[-1]}, duration: {duration_ms} ms, rows: {len(seg)}")

        if duration_ms >= 200:
            filtered.append(seg)

    segments = filtered
    return segments


def compute_trajectory_features(df, user_id):
    """Compute trajectory-level features""" 

    df = df.copy()   
    trajectories = split_trajectory(df)

    all_features = []
    segment_id = 0

    #plt.figure(figsize=(10, 6))


    for df_trajectory in trajectories:
        df_trajectory = df_trajectory.copy()

        # Time total
        time_duration = df_trajectory['client timestamp'].iloc[-1] - df_trajectory['client timestamp'].iloc[0]
        print(f"Segment {segment_id}: duration {time_duration} ms, rows: {len(df_trajectory)}")

        # Velocity stats
        median_vel = df_trajectory['vel'].median()
        std_vel = df_trajectory['vel'].std()
        p10_vel = df_trajectory['vel'].quantile(0.1)
        p90_vel = df_trajectory['vel'].quantile(0.9)
    
        # Acceleration stats
        std_acc = df_trajectory['acc'].std()
        mean_abs_acc = df_trajectory['acc'].abs().mean()

        # Jerk stats
        mean_jerk = df_trajectory['jerk'].mean()
        std_jerk = df_trajectory['jerk'].std()

        # Timing stats
        std_dt = df_trajectory['dt'].std()
        
        std_angle = df_trajectory['angle'].std()
        mean_angle_change = df_trajectory['angle_change'].abs().mean()



        # Path length 
        path_length = df_trajectory['dist'].sum()
        
        # Net displacement 
        start_x, start_y = df_trajectory.iloc[0][['x', 'y']]
        end_x, end_y = df_trajectory.iloc[-1][['x', 'y']]
        net_displacement = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        
        # Straightness ratio
        straightness_ratio = net_displacement / path_length if path_length > 0 else 0
        
        # sign changes v dx/dy
        dx_signs = np.sign(df_trajectory['dx'])
        dy_signs = np.sign(df_trajectory['dy'])
        num_direction_changes = (
            (dx_signs.diff() != 0).sum() + 
            (dy_signs.diff() != 0).sum()
        )

       

        all_features.append({
            'user_id': user_id,
            'segment_id': segment_id,
            'num_events': len(df_trajectory),
            'straightness_ratio': straightness_ratio,
            'num_direction_changes': num_direction_changes,
            'median_vel': median_vel,
            'std_vel': std_vel,
            'p10_vel': p10_vel,
            'p90_vel': p90_vel,
            'std_acc': std_acc,
            'mean_abs_acc': mean_abs_acc,
            'mean_jerk': mean_jerk,
            'std_jerk': std_jerk,
            'std_dt' : std_dt,
            'std_angle': std_angle,
            'mean_angle_change': mean_angle_change,
        })

        segment_id += 1



    # convert to DataFrame
    features_df = pd.DataFrame(all_features)

    features_df.to_csv("splitted_features.csv", index=False)

    return features_df
    



 
#  ---------------- DEBUG ----------------
def write_all_segments_to_csv(segments):

    combined = []
    for i, seg in enumerate(segments):
        seg_copy = seg.copy()
        seg_copy['segment_id'] = i  # segment index
        combined.append(seg_copy)

    if combined:
        out_df = pd.concat(combined, ignore_index=True)
        # put segment_id as first column
        cols = ['segment_id'] + [c for c in out_df.columns if c != 'segment_id']
        out_df = out_df[cols]
        out_df.to_csv("trajectory_segments.csv", index=False)
    else:
        # write empty file with no rows
        pd.DataFrame().to_csv("trajectory_segments.csv", index=False)


def plot_trajectories(df_trajectory):
    color = 'blue' if df_trajectory['state'].iloc[0] == 'Move' else 'red'
    # X osa = lokálny index trajektórie
    x = range(len(df_trajectory))

    plt.plot(x, df_trajectory['vel'], color=color, alpha=0.2)

    
    plt.ylabel("Rýchlosť")
    plt.title("Rýchlosti trajektórií myši")
    plt.grid(True)
    plt.legend(["Move", "Drag"])

    plt.show()