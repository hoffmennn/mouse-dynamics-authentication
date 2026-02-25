import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample

# 1. Načítanie dát
df_train = pd.read_csv("train_data_sw12.csv")
df_test = pd.read_csv("test_data_sw12.csv")

# Oprava preklepu z "culumns_to_drop" :)
columns_to_drop = ["user_id", "csv_file", "window_id"] 

# Získanie všetkých unikátnych používateľov z trénovacej množiny
all_users = sorted(df_train["user_id"].unique())

# Hranica pre "neprípustné" EER (nastav podľa tvojej definície, napr. 15% = 0.15)
MAX_ACCEPTABLE_EER = 0.15

# Zoznam pre ukladanie výsledkov každého používateľa
global_results = []

print(f"Začínam trénovanie modelov pre {len(all_users)} používateľov...\n")

# 2. Hlavný cyklus pre každého používateľa
for target_user_id in all_users:
    # Vytvorenie kópií pre aktuálnu iteráciu
    df_train_user = df_train.copy()
    df_test_user = df_test.copy()
    
    # Vytvorenie binárnych labelov pre aktuálneho používateľa
    df_train_user["label"] = (df_train_user.user_id == target_user_id).astype(int)
    df_test_user["label"] = (df_test_user.user_id == target_user_id).astype(int)
    
    # --- UNDERSAMPLING 
    df_train_genuine = df_train_user[df_train_user["label"] == 1]
    df_train_impostor = df_train_user[df_train_user["label"] == 0]

    df_train_impostor_downsampled = resample(
        df_train_impostor,
        replace=False,
        n_samples=len(df_train_genuine),
        random_state=42,
    )
    
    # Ponechané tvoje pôvodné spájanie (kde sa nepoužil downsampled df). 
    # Ak by si to chcel niekedy aktivovať, prepíš 'df_train_impostor' na 'df_train_impostor_downsampled'
    df_train_balanced = pd.concat([df_train_genuine, df_train_impostor]).sample(frac=1, random_state=42).reset_index(drop=True)
    # -----------------------------------------------------------

    # Príprava matíc pre model
    x_train_balanced = df_train_balanced.drop(columns=columns_to_drop + ["label"], errors='ignore')
    y_train_balanced = df_train_balanced["label"]

    x_test = df_test_user.drop(columns=columns_to_drop + ["label"], errors='ignore')
    y_test = df_test_user["label"]
    
    # Prevencia proti chybám: preskočíme používateľa, ak v teste alebo tréningu nemá žiadne pozitívne vzorky
    if y_test.sum() == 0 or y_train_balanced.sum() == 0:
        print(f"Používateľ {target_user_id:3d} | PRESKOČENÝ (chýbajú vzorky)")
        continue

    # 3. Trénovanie Random Forest
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        random_state=42,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    rf.fit(x_train_balanced, y_train_balanced)

    # 4. Predikcia a výpočet metrík
    y_proba = rf.predict_proba(x_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    EER = fpr[eer_index]
    
    # Uloženie výsledkov do zoznamu
    global_results.append({
        "user_id": target_user_id,
        "AUC": roc_auc,
        "EER": EER
    })

    # Priebežný výpis pre kontrolu
    print(f"Používateľ {target_user_id:3d} | AUC: {roc_auc:.4f} | EER: {EER:.4f}")

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
print(" GLOBÁLNE VYHODNOTENIE (120 POUŽÍVATEĽOV)")
print(f"{'='*45}")
print(f"Počet úspešne vyhodnotených používateľov: {total_users_evaluated}")
print(f"{'-'*45}")
print(f"AUC -> Priemer: {avg_auc:.4f} | Min: {min_auc:.4f} | Max: {max_auc:.4f}")
print(f"EER -> Priemer: {avg_eer:.4f} | Min: {min_eer:.4f} | Max: {max_eer:.4f}")
print(f"{'-'*45}")
print(f"Používatelia s neprípustným EER (> {MAX_ACCEPTABLE_EER:.2f}): {unacceptable_eer_count} ({unacceptable_eer_count/total_users_evaluated:.1%})")
print(f"{'='*45}")