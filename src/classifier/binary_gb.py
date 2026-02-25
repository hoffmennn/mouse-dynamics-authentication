import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc

# 1. Načítanie dát
df_train = pd.read_csv("train_data_sw12.csv")
df_test = pd.read_csv("test_data_sw12.csv")

columns_to_drop = ["user_id", "csv_file", "window_id"]

all_users = sorted(df_train["user_id"].unique())

MAX_ACCEPTABLE_EER = 0.15
global_results = []

print(f"Začínam trénovanie modelov pre {len(all_users)} používateľov...\n")

for target_user_id in all_users:

    df_train_user = df_train.copy()
    df_test_user = df_test.copy()
    
    df_train_user["label"] = (df_train_user.user_id == target_user_id).astype(int)
    df_test_user["label"] = (df_test_user.user_id == target_user_id).astype(int)

    # Použijeme CELÚ impostor množinu
    df_train_full = df_train_user.sample(frac=1, random_state=42).reset_index(drop=True)

    x_train = df_train_full.drop(columns=columns_to_drop + ["label"], errors='ignore')
    y_train = df_train_full["label"]

    x_test = df_test_user.drop(columns=columns_to_drop + ["label"], errors='ignore')
    y_test = df_test_user["label"]

    # Kontrola dát
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count

    if y_test.sum() == 0 or pos_count == 0:
        print(f"Používateľ {target_user_id:3d} | PRESKOČENÝ (chýbajú vzorky)")
        continue

    # Výpočet váhy
    scale_pos_weight = neg_count / pos_count

    # 3. XGBoost model
    xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        device="cuda"
    )

    xgb.fit(x_train, y_train)

    # 4. Predikcia
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

    print(f"Používateľ {target_user_id:3d} | "
          f"pos:{int(pos_count)} neg:{int(neg_count)} "
          f"| AUC: {roc_auc:.4f} | EER: {EER:.4f}")

# 5. GLOBÁLNE VYHODNOTENIE
df_results = pd.DataFrame(global_results)

avg_auc = df_results["AUC"].mean()
max_auc = df_results["AUC"].max()
min_auc = df_results["AUC"].min()

avg_eer = df_results["EER"].mean()
max_eer = df_results["EER"].max()
min_eer = df_results["EER"].min()

unacceptable_eer_count = (df_results["EER"] > MAX_ACCEPTABLE_EER).sum()
total_users_evaluated = len(df_results)

print(f"\n{'='*45}")
print(" GLOBÁLNE VYHODNOTENIE")
print(f"{'='*45}")
print(f"Počet úspešne vyhodnotených používateľov: {total_users_evaluated}")
print(f"{'-'*45}")
print(f"AUC -> Priemer: {avg_auc:.4f} | Min: {min_auc:.4f} | Max: {max_auc:.4f}")
print(f"EER -> Priemer: {avg_eer:.4f} | Min: {min_eer:.4f} | Max: {max_eer:.4f}")
print(f"{'-'*45}")
print(f"Používatelia s neprípustným EER (> {MAX_ACCEPTABLE_EER:.2f}): "
      f"{unacceptable_eer_count} "
      f"({unacceptable_eer_count/total_users_evaluated:.1%})")
print(f"{'='*45}")