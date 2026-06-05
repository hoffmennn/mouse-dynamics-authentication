import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve, auc


def remove_users(df_train, df_test, users_to_remove):

    before_train = len(df_train)
    before_test = len(df_test)
    
    
    df_train_cleaned = df_train[~df_train["user_id"].isin(users_to_remove)].copy()
    df_test_cleaned = df_test[~df_test["user_id"].isin(users_to_remove)].copy()
    
    print(f"--- ČISTENIE DÁT ---")
    print(f"Odstraňujem používateľov: {users_to_remove}")
    print(f"Train: odstránených {before_train - len(df_train_cleaned)} riadkov.")
    print(f"Test:  odstránených {before_test - len(df_test_cleaned)} riadkov.")
    print(f"--------------------\n")
    
    return df_train_cleaned, df_test_cleaned

# 1. Načítanie dát
df_train = pd.read_csv("train-ws22-st4.csv")
df_test = pd.read_csv("test-ws22-st4 copy.csv")

users_to_drop = [4, 103]
#users_to_drop = [4] users_to_drop = [4, 103, 60, 99, 36]
df_train, df_test = remove_users(df_train, df_test, users_to_drop)

columns_to_drop = ["user_id", "csv_file", "window_id", "std_angle_std", "std_angle_mean", "scattering_coefficient_norm_std", "scattering_coefficient_norm_mean", "tcm_norm_std", "std_dt_std"]
#columns_to_drop = ["user_id", "csv_file", "window_id"]

all_users = sorted(df_train["user_id"].unique())

MAX_ACCEPTABLE_EER = 0.15
global_results = []

print(f"training started...  {len(all_users)} users...\n")

for target_user_id in all_users:

    df_train_user = df_train.copy()
    df_test_user = df_test.copy()
    
    df_train_user["label"] = (df_train_user.user_id == target_user_id).astype(int)
    df_test_user["label"] = (df_test_user.user_id == target_user_id).astype(int)

    df_train_full = df_train_user.sample(frac=1, random_state=42).reset_index(drop=True)

    x_train = df_train_full.drop(columns=columns_to_drop + ["label"], errors='ignore')
    y_train = df_train_full["label"]

    x_test = df_test_user.drop(columns=columns_to_drop + ["label"], errors='ignore')
    y_test = df_test_user["label"]

    # Kontrola dát
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count

    if y_test.sum() == 0 or pos_count == 0:
        print(f"user {target_user_id:3d} | too few samples")
        continue

    # Výpočet váhy
    scale_pos_weight = neg_count / pos_count

    # 3. XGBoost model
    lgbm = LGBMClassifier(
        # === POČET STROMOV ===
        n_estimators=500,

        # === UČIACA MIERA ===
        learning_rate=0.03,

        # === ŠTRUKTÚRA STROMOV (leaf-wise) ===
        num_leaves=15,              # Hlavný parameter komplexnosti v LightGBM.
                                    # 15 je dobrý kompromis — dosť na zachytenie
                                    # interakcií, málo na overfitting.

        max_depth=4,                # Bezpečnostný limit. Pri 15 listoch reálne
                                    # dosiahneš hĺbku 3-4, takže toto len
                                    # zabraňuje extrémnym prípadom.

        # === MINIMÁLNE VZORKY ===
        min_child_samples=10,       # ZVÝŠENÉ z 5. Minimálny počet vzoriek v liste.
                                    # Pri ~30-50 pozitívnych vzorkách na používateľa
                                    # je 10 konzervatívne ale bezpečné.

        min_child_weight=0.01,      # ZVÝŠENÉ z 0.001. Stále nízke, ale nie
                                    # prakticky nulové.

        # === STOCHASTICKÉ VZORKOVANIE ===
        subsample=0.75,             # Znížené z 0.8
        subsample_freq=1,           # Nutné, aby subsample fungoval
        colsample_bytree=0.65,      # Znížené z 0.8 — väčšia diverzita stromov

        # === REGULARIZÁCIA ===
        reg_alpha=0.5,              # ZVÝŠENÉ — L1 regularizácia
        reg_lambda=2.0,             # ZVÝŠENÉ — L2 regularizácia
        min_gain_to_split=0.1,      # NOVÉ — ekvivalent gamma v XGBoost.
                                    # Split sa vykoná len ak prinesie gain > 0.1

        # === NEVYVÁŽENOSŤ ===
        scale_pos_weight=scale_pos_weight,

        # === OSTATNÉ ===
        objective="binary",
        random_state=42,
        n_jobs=-1,
        importance_type='gain',
        verbosity=-1
    )

    lgbm.fit(x_train, y_train)

    # 4. Predikcia
    y_proba = lgbm.predict_proba(x_test)[:, 1]

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

print("evaluation")
print(f"{'.'*45}")
print(f"num users: {total_users_evaluated}")
print(f"AUC   avg: {avg_auc:.4f} | min: {min_auc:.4f} | max: {max_auc:.4f}")
print(f"EER   avg: {avg_eer:.4f} | min: {min_eer:.4f} | max: {max_eer:.4f}")
print(f"{'.'*45}")
print(f"users with too high EER (> {MAX_ACCEPTABLE_EER:.2f}): "
      f"{unacceptable_eer_count} "
      f"({unacceptable_eer_count/total_users_evaluated:.1%})")
