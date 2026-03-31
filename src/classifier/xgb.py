import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# load data
df_train = pd.read_csv("train-ws15-st3.csv")
df_test = pd.read_csv("test-ws15-st3.csv")

columns_to_drop = ["user_id", "csv_file", "window_id"]

all_users = sorted(df_train["user_id"].unique())

MAX_ACCEPTABLE_EER = 0.15
global_results = []

print(f"training started...  {len(all_users)} users...\n")

for target_user_id in all_users:

    # label user
    df_train_user = df_train.copy()
    df_test_user = df_test.copy()
    
    df_train_user["label"] = (df_train_user.user_id == target_user_id).astype(int)
    df_test_user["label"] = (df_test_user.user_id == target_user_id).astype(int)

    x_train_all = df_train_user.drop(columns=columns_to_drop + ["label"], errors='ignore')
    y_train_all = df_train_user["label"]

    # validation set
    x_train_real, x_val, y_train_real, y_val = train_test_split(
        x_train_all, y_train_all, 
        test_size = 0.15,            
        stratify=y_train_all,     
        random_state = 5
    )

    x_test = df_test_user.drop(columns=columns_to_drop + ["label"], errors='ignore')
    y_test = df_test_user["label"]

    # scaling weights
    pos_count = y_train_real.sum()
    neg_count = len(y_train_real) - pos_count
    
    if y_test.sum() == 0 or pos_count == 0:
        continue

    scale_pos_weight = neg_count / pos_count

    
    xgb = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",   
        early_stopping_rounds=50, 
        random_state=5,
        n_jobs=-1,
        tree_method="hist",
        device="cuda"
    )

    #training
    xgb.fit(
        x_train_real, y_train_real,
        eval_set=[(x_val, y_val)],
        verbose=False              
    )

    print(f"user: {target_user_id:3d}  best iteration: {xgb.best_iteration}")

    # prediction
    y_proba = xgb.predict_proba(x_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)



    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    EER = fpr[eer_index]
    
    global_results.append({
        "user_id": target_user_id,
        "AUC": roc_auc,
        "EER": EER
    })

    print(f"user {target_user_id:3d} | "
          f"pos:{int(pos_count)} neg:{int(neg_count)} "
          f"| AUC: {roc_auc:.4f} | EER: {EER:.4f}")

# evaluation
df_results = pd.DataFrame(global_results)

avg_auc = df_results["AUC"].mean()
max_auc = df_results["AUC"].max()
min_auc = df_results["AUC"].min()

avg_eer = df_results["EER"].mean()
max_eer = df_results["EER"].max()
min_eer = df_results["EER"].min()

unacceptable_eer_count = (df_results["EER"] > MAX_ACCEPTABLE_EER).sum()
total_users_evaluated = len(df_results)

print("evaluation")
print(f"{'.'*45}")
print(f"num users: {total_users_evaluated}")
print(f"AUC   avg: {avg_auc:.4f} | min: {min_auc:.4f} | max: {max_auc:.4f}")
print(f"EER   avg: {avg_eer:.4f} | min: {min_eer:.4f} | max: {max_eer:.4f}")
print(f"{'.'*45}")
print(f"users with too high EER (> {MAX_ACCEPTABLE_EER:.2f}): "
      f"{unacceptable_eer_count} "
      f"({unacceptable_eer_count/total_users_evaluated:.1%})")
