import pandas as pd
from pathlib import Path

import click_features as cf
import trajectory_features as tf
import event_features as ef

ROOT_DIR = Path(__file__).resolve().parents[2]
#data_file = ROOT_DIR / "data" / "sapimouse" / "sapimouse" / "user55" / "session_2020_05_14_3min.csv"

def load_session_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def extract_user_features(csv_path, user_id):
    

    df = load_session_data(csv_path)
    
    #print(f"loaded {len(df)} events")
    #print(f"time duration: {df['client timestamp'].min()} - {df['client timestamp'].max()}")

    df = ef.compute_event_features(df)

    click_features = cf.compute_click_features(df)
    trajectory_features = tf.compute_trajectory_features(df, user_id)


    #append click_features to all trajectories
    for key, value in click_features.items():
        trajectory_features[key] = value

    #debug
    #trajectory_features.to_csv("user_full_features.csv", index=False)

    return trajectory_features



if __name__ == "__main__":

    USERS_ROOT = ROOT_DIR / "data" / "sapimouse" / "sapimouse"

    all_features = []  

    
    for user_dir in USERS_ROOT.iterdir():
        if not user_dir.is_dir():
            continue
        
     
        # extract user_id from folder name user55 -> 55
        try:
            user_id = int(user_dir.name.replace("user", ""))
        except:
            print(f"unable to load: {user_dir.name}  ")
            continue

        print(f"\nprocessing user {user_id} ...")

        
        for csv_path in user_dir.glob("session_*_1min.csv"):
            print(f"-file: {csv_path.name}")

            try:
                features_df = extract_user_features(csv_path, user_id)
                features_df["csv_file"] = csv_path.name 
                all_features.append(features_df)
            except Exception as e:
                print(f"  !!! error in processing {csv_path.name}: {e}")

    # save all features
    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        final_df.to_csv("TEST_FEATURES.csv", index=False)
        print("\ndataset saved")
    else:
        print("no features extracted")