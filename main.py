import pandas as pd
from pathlib import Path
from src.processing import features

ROOT_DIR = Path(__file__).parent 
USERS_ROOT = ROOT_DIR / "data" / "sapimouse" / "sapimouse"

def get_user_id_from_path(path):
    """extract user_id from folder name"""
    try:
        return int(path.name.replace("user", ""))
    except ValueError:
        return None
    
def main():
    all_features = []

    for user_dir in USERS_ROOT.iterdir():
        if not user_dir.is_dir():
            continue

        user_id = get_user_id_from_path(user_dir)
        if user_id is None:
            continue

        print(f"\nProcessing user {user_id} ...")

        
        for csv_path in user_dir.glob("session_*_1min.csv"):
            print(f" - file: {csv_path.name}")

            try:
                
                df = features.load_session_data(csv_path)
                
                features_df = features.extract_user_features(df, user_id)
                
                features_df["csv_file"] = csv_path.name
                all_features.append(features_df)

            except Exception as e:
                print(f"   !!! error in processing {csv_path.name}: {e}")


    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        final_df.to_csv("test_data.csv", index=False)
        print("\ndataset saved")

    else:
        print("No features extracted.")

if __name__ == "__main__":
    main()