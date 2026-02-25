import pandas as pd

from . import event_features as ef
from . import click_features as cf
from . import trajectory_features as tf 


def load_session_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def extract_user_features(df, user_id):
    """Extract features pipeline for a single user session"""

    df = ef.compute_event_features(df)

    #click_features = cf.compute_click_features(df)
    trajectory_features = tf.compute_trajectory_features(df, user_id)

    #for key, value in click_features.items():
    #    trajectory_features[key] = value

    #debug
    #trajectory_features.to_csv("user_full_features.csv", index=False)

    return trajectory_features