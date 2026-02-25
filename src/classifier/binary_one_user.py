import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

df_train = pd.read_csv("train_data_sw.csv")
df_test = pd.read_csv("test_data_sw.csv")

TARGET_USER_ID = 27

df_train["label"] = (df_train.user_id == TARGET_USER_ID).astype(int)
df_test["label"] = (df_test.user_id == TARGET_USER_ID).astype(int)

mouse_features = ["num_clicks","mean_click_duration","std_click_duration","mean_inter_click_time","std_inter_click_time"]
#df_train.drop(columns=mouse_features, inplace=True)
#df_test.drop(columns=mouse_features, inplace=True)

culumns_to_drop = ["user_id", "csv_file", "window_id"] 

x_train = df_train.drop(columns=culumns_to_drop + ["label"])
y_train = df_train["label"]

x_test = df_test.drop(columns=culumns_to_drop + ["label"])
y_test = df_test["label"]



# under-sampling impostor class
df_train_genuine = df_train[df_train["label"] == 1]
df_train_impostor = df_train[df_train["label"] == 0]

df_train_impostor_downsampled = resample(
    df_train_impostor,
    replace=False,
    n_samples=len(df_train_genuine),
    random_state=42,
)

print(f"genuine: {len(df_train_genuine)}  impostor (downsampled): {len(df_train_impostor)}")


df_train_balanced = pd.concat([df_train_genuine, df_train_impostor]).sample(frac=1, random_state=42).reset_index(drop=True)

x_train_balanced = df_train_balanced.drop(columns=culumns_to_drop + ["label"])
y_train_balanced = df_train_balanced["label"]





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

y_pred = rf.predict(x_test)
y_proba = rf.predict_proba(x_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
print(f"AUC = {roc_auc:.4f}")


fnr = 1 - tpr

eer_index = np.nanargmin(np.abs(fnr - fpr))
EER = fpr[eer_index]

optimal_threshold = thresholds[eer_index]
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)




# --- 1. ZÁKLADNÉ METRIKY ---
print(f"{'='*30}")
print(f" VYHODNOTENIE PRE POUŽÍVATEĽA: {TARGET_USER_ID}")
print(f"{'='*30}")
print(f"AUC: {roc_auc:.4f}")
print(f"EER: {EER:.4f}")
print(f"Optimálny prah (EER): {optimal_threshold:.4f}")
print(f"{'-'*30}")

# --- 2. POROVNANIE PRAHOV (0.5 vs OPTIMAL) ---

def print_metrics(y_true, y_p, title):
    tn, fp, fn, tp = confusion_matrix(y_true, y_p).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    print(f"\n[{title}]")
    print(f"Confusion Matrix: TP:{tp}, FP:{fp}, TN:{tn}, FN:{fn}")
    print(f"FAR (Falošné prijatie): {far:.4%}")
    print(f"FRR (Falošné odmietnutie): {frr:.4%}")

# Pôvodný prah 0.5 (štandard v RF)
print_metrics(y_test, y_pred, "ŠTANDARDNÝ PRAH 0.5")

# Optimálny prah z EER
print_metrics(y_test, y_pred_optimal, "OPTIMÁLNY PRAH (EER)")

# --- 3. FEATURE IMPORTANCE (Dôležitosť čŕt) ---
importances = rf.feature_importances_
features = x_train_balanced.columns
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

print(f"\n{'-'*30}")
print("most important features:")
print(feat_imp.head(10))
print(f"{'='*30}")