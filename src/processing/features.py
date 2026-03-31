import pandas as pd

from . import event_features as ef
from . import click_features as cf
from . import trajectory_features as tf 
from . import segment_processing as sp

def load_session_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def extract_user_features(df, user_id):

    df = ef.compute_event_features(df)

    df_features = tf.compute_trajectory_features(df, user_id)
    
    df_features_sliding_window = sp.apply_sliding_window(df_features, user_id, 22, 3)
    
    #debug
    #trajectory_features.to_csv("user_full_features.csv", index=False)

    return df_features_sliding_window