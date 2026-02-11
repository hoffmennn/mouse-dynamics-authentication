import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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