import pandas as pd
import numpy as np
import os
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
from scipy.interpolate import UnivariateSpline

def save_model(model, user_id, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    booster = model.get_booster()
    booster.feature_names = list(model.feature_names_in_)
    path = os.path.join(output_dir, f"user_{user_id}.json")
    booster.save_model(path)
    print(f"Model uložený: {path}")

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

def plot_shap_summary(model, x_data, user_id):
    """
    Vypočíta SHAP hodnoty s opravou pre chybu '[5E-1]' a TypeError.
    """
    print(f"Generujem SHAP analýzu pre používateľa {user_id}...")
    
    try:
        # TRIK 1: Manuálne nastavenie base_score na float, aby SHAP nezlyhal na stringu '[5E-1]'
        if hasattr(model, "base_score") and isinstance(model.base_score, str):
            model.base_score = 0.5  # Štandardná hodnota pre binárnu logistiku

        # TRIK 2: Použijeme TreeExplainer priamo na 'booster' objekt, 
        # čo obchádza Scikit-learn wrapper, ktorý robí problémy.
        booster = model.get_booster()
        explainer = shap.TreeExplainer(booster)
        
        # Výpočet SHAP hodnôt (použijeme len x_data)
        shap_values = explainer.shap_values(x_data)

        # Vizualizácia
        plt.figure(figsize=(10, 6))
        
        # Pri binárnej klasifikácii v XGBoost 2.0+ vracajú TreeExplainery 
        # niekedy 2D pole, niekedy 1D. Musíme to ošetriť.
        shap.summary_plot(shap_values, x_data, show=False)
        
        plt.title(f"SHAP Top Features - Používateľ {user_id}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"shap_summary_user_{user_id}.png", dpi=300)
        plt.show()
        plt.close()
        print(f"Graf uložený ako: shap_summary_user_{user_id}.png")

    except Exception as e:
        print(f" SHAP zlyhal aj po opravách: {e}")
        print(" Skús v termináli: pip install xgboost==1.7.6 (staršia stabilnejšia verzia pre SHAP)")

def plot_global_feature_importance(all_importances, feature_names):
    """
    Vykreslí priemernú dôležitosť pre VŠETKY príznaky naprieč všetkými modelmi.
    """
    # 1. Prevod na DataFrame (riadky = používatelia, stĺpce = príznaky)
    df_imp = pd.DataFrame(all_importances, columns=feature_names)
    
    # 2. Výpočet priemeru a zoradenie (od najdôležitejšej po najmenej)
    mean_imp = df_imp.mean().sort_values(ascending=False)
    
    # 3. Dynamické nastavenie výšky grafu (napr. 0.3 palca na každý príznak)
    # Ak máš 50 príznakov, výška bude 15 palcov - to zaručí čitateľnosť.
    plt_height = max(6, len(feature_names) * 0.3)
    
    plt.figure(figsize=(12, plt_height))
    
    # Vykreslenie všetkých príznakov
    sns.barplot(
        x=mean_imp.values, 
        y=mean_imp.index, 
        palette="magma" # Pekný farebný prechod od tmavej po svetlú
    )

    y_positions = np.arange(len(mean_imp))
    x_values = mean_imp.values

    # s = parameter vyhladenia. Čím vyššie číslo, tým hladšia krivka. 
    # Skús začať na s=0.5 alebo s=len(mean_imp)*0.1
    spl = UnivariateSpline(y_positions, x_values, s=0.5) 

    y_smooth = np.linspace(0, len(mean_imp) - 1, 300)
    x_smooth = spl(y_smooth)
    plt.plot(x_smooth, y_smooth, color="#FF0000", linewidth=2.5, label="Trend")
    
    plt.title(f"Kompletná globálna dôležitosť čŕt (Priemer cez {len(all_importances)} používateľov)", fontsize=16)
    plt.xlabel("Priemerný prínos (Gain Importance)", fontsize=12)
    plt.ylabel("Názov biometrického príznaku", fontsize=12)
    
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Uloženie s vysokým rozlíšením
    plt.savefig("global_feature_importance_ALL.png", dpi=300)
    plt.show()

    # Vypísanie kompletného zoznamu do konzoly pre ľahké kopírovanie do Excelu/práce
    print("\n" + "="*50)
    print("KOMPLETNÝ REBRÍČEK DÔLEŽITOSTI (Zoradené):")
    print("="*50)
    for i, (name, val) in enumerate(mean_imp.items(), 1):
        print(f"{i:3d}. {name:30s} | Gain: {val:.6f}")
    print("="*50)

def plot_eer_histogram(df_results, max_acceptable_eer):

    plt.figure(figsize=(10, 6))
    sns.histplot(df_results["EER"], bins=500, kde=True, color="skyblue", edgecolor="black")
    
    # Pridáme čiaru pre priemerné EER
    avg_eer = df_results["EER"].mean()
    plt.axvline(avg_eer, color="red", linestyle="dashed", linewidth=2, label=f"Priemerné EER: {avg_eer:.4f}")
    
    # Pridáme čiaru pre tvoj prah akceptovateľnosti
    plt.axvline(max_acceptable_eer, color="orange", linestyle="dashed", linewidth=2, label=f"Max akceptovateľné: {max_acceptable_eer}")
    
    plt.title("Rozloženie Equal Error Rate (EER) naprieč používateľmi", fontsize=14)
    plt.xlabel("EER (Hodnota)", fontsize=12)
    plt.ylabel("Počet používateľov", fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    
    # Uloženie do súboru
    plt.savefig("eer_histogram.png", dpi=300)
    plt.show()
    plt.close()

def plot_sorted_eer(df_results):
    # Zoradíme dáta od najmenšieho EER
    sorted_eer = df_results["EER"].sort_values().values
    user_index = range(1, len(sorted_eer) + 1)

    plt.figure(figsize=(10, 6))
    plt.scatter(user_index, sorted_eer, color='teal', s=15, label='EER používateľa')
    plt.fill_between(user_index, sorted_eer, color='teal', alpha=0.1)

    plt.axhline(y=0.15, color='orange', linestyle='--', label='Prah 0.15')
    
    plt.title("EER zoradené podľa kvality modelu (1 = najlepší)")
    plt.xlabel("Poradie používateľa")
    plt.ylabel("EER")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_eer_boxplot(df_results, max_acceptable_eer):
    """
    Boxplot pre EER s označením outlierov.
    """
    plt.figure(figsize=(8, 6))
    
    box = plt.boxplot(df_results["EER"], vert=True, patch_artist=True,
                      widths=0.5,
                      boxprops=dict(facecolor='lightblue', color='darkblue'),
                      medianprops=dict(color='red', linewidth=2),
                      whiskerprops=dict(color='darkblue'),
                      capprops=dict(color='darkblue'),
                      flierprops=dict(marker='o', markerfacecolor='orange', 
                                     markersize=8, linestyle='none'))
    
    # Pridaj horizontálnu čiaru pre prah
    plt.axhline(y=max_acceptable_eer, color='orange', linestyle='--', 
                linewidth=2, label=f'Prah: {max_acceptable_eer}')
    
    # Pridaj priemernú hodnotu
    avg_eer = df_results["EER"].mean()
    plt.axhline(y=avg_eer, color='green', linestyle=':', 
                linewidth=2, label=f'Priemer: {avg_eer:.4f}')
    
    plt.ylabel('EER', fontsize=12)
    plt.title('Boxplot EER hodnôt naprieč používateľmi', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig("eer_boxplot.png", dpi=300)
    plt.show()
    plt.close()

def plot_far_frr_eer(
        fpr,
        fnr,
        thresholds,
        eer_threshold,
        eer_value,
        user_id,
        save_path=None
):
    """
    Graf FAR (False Acceptance Rate), FRR (False Rejection Rate)
    a EER bodu pre konkrétneho používateľa.
    """

    # -------------------------------------------------
    # ODSTRÁNENIE inf thresholdov
    # -------------------------------------------------
    valid = np.isfinite(thresholds)

    thresholds_plot = thresholds[valid]
    fpr_plot = fpr[valid]
    fnr_plot = fnr[valid]

    # -------------------------------------------------
    # ZORADENIE thresholdov vzostupne
    # -------------------------------------------------
    order = np.argsort(thresholds_plot)

    thresholds_plot = thresholds_plot[order]
    fpr_plot = fpr_plot[order]
    fnr_plot = fnr_plot[order]

    # -------------------------------------------------
    # VYKRESLENIE
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        thresholds_plot,
        fpr_plot,
        color="#E74C3C",
        linewidth=2,
        label="FAR (False Acceptance Rate)"
    )

    ax.plot(
        thresholds_plot,
        fnr_plot,
        color="#2980B9",
        linewidth=2,
        label="FRR (False Rejection Rate)"
    )

    # EER threshold
    ax.axvline(
        eer_threshold,
        color="gray",
        linestyle="--",
        linewidth=1.4,
        label=f"EER prah = {eer_threshold:.4f}"
    )

    # EER bod
    ax.scatter(
        [eer_threshold],
        [eer_value],
        color="black",
        zorder=5,
        s=70,
        label=f"EER = {eer_value:.4f}"
    )

    # horizontálna čiara
    ax.axhline(
        eer_value,
        color="gray",
        linestyle=":",
        linewidth=1,
        alpha=0.6
    )

    ax.set_xlabel("Rozhodovací prah (threshold)", fontsize=12)
    ax.set_ylabel("Chybovosť", fontsize=12)

    ax.set_title(
        f"FAR / FRR / EER – Používateľ {user_id}",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()

    if save_path is None:
        save_path = f"far_frr_eer_user_{user_id}.png"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Graf uložený: {save_path}")


# -------------- config --------------


df_train = pd.read_csv("train-ws22-st4.csv")
df_test = pd.read_csv("test-ws22-st4.csv")
#df_train = pd.read_csv("train_data_sw15.csv")
#df_test = pd.read_csv("test_data_sw15.csv")


users_to_drop = [4, 103]
df_train, df_test = remove_users(df_train, df_test, users_to_drop)
# window_id,user_id,num_events_mean,num_events_std,median_vel_mean,median_vel_std,std_vel_mean,std_vel_std,p10_vel_mean,p10_vel_std,p90_vel_mean,p90_vel_std,skew_vel_mean,skew_vel_std,kurtosis_vel_mean,kurtosis_vel_std,std_acc_mean,std_acc_std,mean_abs_acc_mean,mean_abs_acc_std,skew_acc_mean,skew_acc_std,kurtosis_acc_mean,kurtosis_acc_std,mean_jerk_mean,mean_jerk_std,std_jerk_mean,std_jerk_std,skew_jerk_mean,skew_jerk_std,kurtosis_jerk_mean,kurtosis_jerk_std,std_dt_mean,std_dt_std,std_angle_mean,std_angle_std,mean_angle_change_mean,mean_angle_change_std,straightness_ratio_mean,straightness_ratio_std,num_direction_changes_mean,num_direction_changes_std,tcm_norm_mean,tcm_norm_std,scattering_coefficient_norm_mean,scattering_coefficient_norm_std,sum_angle_change_mean,sum_angle_change_std,num_pauses_mean,num_pauses_std,time_duration_mean,time_duration_std,click_duration_mean,click_duration_std,csv_file
''' 
columns_to_drop = ["user_id", "csv_file", "window_id", 
                   "std_angle_std", "std_angle_mean", "scattering_coefficient_norm_std", 
                   "scattering_coefficient_norm_mean", "tcm_norm_std", "std_dt_std" ,
                    "num_events_std", "median_vel_std", "std_vel_std", "p10_vel_std", 
                    "p90_vel_std", "skew_vel_std", "kurtosis_vel_std", "std_acc_std", 
                    "mean_abs_acc_std", "skew_acc_std", "kurtosis_acc_std", "mean_jerk_std", 
                    "std_jerk_mean", "std_jerk_std", "skew_jerk_std", "kurtosis_jerk_std", 
                     "mean_angle_change_std", 
                    "straightness_ratio_std", "num_direction_changes_std",
                     "sum_angle_change_std", 
                    "num_pauses_std", "time_duration_std", "click_duration_std"

                    "sum_angle_change_mean", "p10_vel_mean", "p90_vel_mean", "median_vel_mean",
                    "skew_jerk_mean", "num_direction_changes_mean", "mean_angle_change_mean",
                    "skew_acc_mean", "kurtosis_acc_mean", "mean_jerk_mean",
                    "straightness_ratio_mean", "tcm_norm_mean", "kurtosis_jerk_mean"
                     ]
'''
columns_to_drop = ["user_id", "csv_file", "window_id"]

#columns_to_drop = ["user_id", "csv_file", "window_id",
 #                  "std_angle_std", "std_angle_mean", "scattering_coefficient_norm_std", 
#                  "scattering_coefficient_norm_mean", "tcm_norm_std", "std_dt_std" ,]
all_users = sorted(df_train["user_id"].unique())

users_for_xai = [all_users[0], all_users[1]]

users_for_far_frr = []

# ---------------------------



MAX_ACCEPTABLE_EER = 0.15
global_results = []
all_importances = []
for target_user_id in all_users:

    #if target_user_id != 15 and target_user_id != 80 and target_user_id != 90:
    #    continue

    df_train_user = df_train.copy()
    df_test_user = df_test.copy()
    
    df_train_user["label"] = (df_train_user.user_id == target_user_id).astype(int)
    df_test_user["label"] = (df_test_user.user_id == target_user_id).astype(int)


    df_train_full = df_train_user.sample(frac=1, random_state=42).reset_index(drop=True)

    x_train = df_train_full.drop(columns=columns_to_drop + ["label"], errors='ignore')
    y_train = df_train_full["label"]

    x_test = df_test_user.drop(columns=columns_to_drop + ["label"], errors='ignore')
    y_test = df_test_user["label"]

  
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count

    if y_test.sum() == 0 or pos_count == 0:
        continue

    scale_pos_weight = neg_count / pos_count

    xgb = XGBClassifier(
        n_estimators=400,           
        learning_rate=0.025,        
        max_depth=4,                      
        min_child_weight=5,         
        subsample=0.75,             
        colsample_bytree=0.6,       
        reg_alpha=0.5,              
        reg_lambda=3.0,             
        gamma=0.3,                  
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=15,
        n_jobs=-1,
        tree_method="hist",
    )

    xgb.fit(x_train, y_train)
    save_model(xgb, target_user_id) 
    importances = xgb.feature_importances_
    all_importances.append(importances)

    # prediction
    y_proba = xgb.predict_proba(x_test)[:, 1]
    

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    EER = fpr[eer_index]

    # Výpočet prahu pre EER
    eer_threshold = thresholds[eer_index]
    
    # Predikcia tried pre ďalšie metriky (používame EER threshold)
    y_pred = (y_proba >= eer_threshold).astype(int)
    
    # Výpočet accuracy, precision, recall
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    global_results.append({
        "user_id": target_user_id,
        "AUC": roc_auc,
        "EER": EER,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall
    })

    print(f"Používateľ {target_user_id:3d} | "
          f"pos:{int(pos_count)} neg:{int(neg_count)} "
          f"| AUC: {roc_auc:.4f} | EER: {EER:.6f}")

    if target_user_id in users_for_far_frr:
        plot_far_frr_eer(
            fpr=fpr,
            fnr=fnr,
            thresholds=thresholds,
            eer_threshold=eer_threshold,
            eer_value=EER,
            user_id=target_user_id,
            save_path=f"far_frr_eer_user_{target_user_id}.png",
        )

    #if target_user_id in users_for_xai:
    #    plot_shap_summary(xgb, x_test, target_user_id)



# 5. GLOBÁLNE VYHODNOTENIE
df_results = pd.DataFrame(global_results)
# Výpočet štandardných odchýlok a mediánov
std_auc = df_results["AUC"].std()
median_auc = df_results["AUC"].median()

std_eer = df_results["EER"].std()
median_eer = df_results["EER"].median()

avg_accuracy = df_results["Accuracy"].mean()
std_accuracy = df_results["Accuracy"].std()
median_accuracy = df_results["Accuracy"].median()

avg_precision = df_results["Precision"].mean()
std_precision = df_results["Precision"].std()
median_precision = df_results["Precision"].median()

avg_recall = df_results["Recall"].mean()
std_recall = df_results["Recall"].std()
median_recall = df_results["Recall"].median()

avg_auc = df_results["AUC"].mean()
max_auc = df_results["AUC"].max()
min_auc = df_results["AUC"].min()

avg_eer = df_results["EER"].mean()
max_eer = df_results["EER"].max()
min_eer = df_results["EER"].min()

unacceptable_eer_count = (df_results["EER"] > MAX_ACCEPTABLE_EER).sum()
total_users_evaluated = len(df_results)

print(" evaluation ")
print("."*70)
print(f"num users: {total_users_evaluated}")
print(f"{'-'*70}")
print(f"{'-':<15} | {'avg':>8} | {'std':>8} | {'med':>8} | {'min':>8} | {'max':>8}")
print(f"{'-'*70}")
print(f"{'AUC':<15} | {avg_auc:8.4f} | {std_auc:8.4f} | {median_auc:8.4f} | {min_auc:8.4f} | {max_auc:8.4f}")
print(f"{'EER':<15} | {avg_eer:8.4f} | {std_eer:8.4f} | {median_eer:8.4f} | {min_eer:8.4f} | {max_eer:8.4f}")
print(f"{'Accuracy':<15} | {avg_accuracy:8.4f} | {std_accuracy:8.4f} | {median_accuracy:8.4f} | {df_results['Accuracy'].min():8.4f} | {df_results['Accuracy'].max():8.4f}")
print(f"{'Precision':<15} | {avg_precision:8.4f} | {std_precision:8.4f} | {median_precision:8.4f} | {df_results['Precision'].min():8.4f} | {df_results['Precision'].max():8.4f}")
print(f"{'Recall':<15} | {avg_recall:8.4f} | {std_recall:8.4f} | {median_recall:8.4f} | {df_results['Recall'].min():8.4f} | {df_results['Recall'].max():8.4f}")
print(f"{'-'*70}")
print(f"Používatelia s neprípustným EER (> {MAX_ACCEPTABLE_EER:.2f}): "
      f"{unacceptable_eer_count} ({unacceptable_eer_count/total_users_evaluated:.1%})")
print("="*70)

plot_global_feature_importance(all_importances, x_train.columns)
#plot_eer_histogram(df_results, MAX_ACCEPTABLE_EER)
#plot_sorted_eer(df_results)
plot_eer_boxplot(df_results, MAX_ACCEPTABLE_EER)  