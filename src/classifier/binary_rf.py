import pandas as pd
import numpy as np

import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

def remove_users(df_train, df_test, users_to_remove):
    """
    Odstráni špecifikovaných používateľov z trénovacej a testovacej množiny.
    """
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
    Vypočíta SHAP hodnoty pre Random Forest model.
    """
    print(f"Generujem SHAP analýzu pre používateľa {user_id}...")
    
    try:
        # Pre Random Forest používame TreeExplainer
        explainer = shap.TreeExplainer(model)
        
        # Výpočet SHAP hodnôt
        shap_values = explainer.shap_values(x_data)
        
        # Vizualizácia
        plt.figure(figsize=(10, 6))
        
        # Pri binárnej klasifikácii v RF môže byť shap_values list alebo array
        # Ak je to list, berieme druhý element (pozitívna trieda)
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values
            
        shap.summary_plot(shap_values_to_plot, x_data, show=False)
        
        plt.title(f"SHAP Top Features - Používateľ {user_id}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"shap_summary_user_{user_id}_RF.png", dpi=300)
        plt.show()
        plt.close()
        print(f"Graf uložený ako: shap_summary_user_{user_id}_RF.png")

    except Exception as e:
        print(f"⚠ SHAP zlyhal: {e}")
        print("SHAP pre Random Forest môže byť pomalý pri veľkých datasetoch.")

def plot_global_feature_importance(all_importances, feature_names):
    """
    Vykreslí priemernú dôležitosť pre VŠETKY príznaky naprieč všetkými modelmi.
    """
    # 1. Prevod na DataFrame (riadky = používatelia, stĺpce = príznaky)
    df_imp = pd.DataFrame(all_importances, columns=feature_names)
    
    # 2. Výpočet priemeru a zoradenie (od najdôležitejšej po najmenej)
    mean_imp = df_imp.mean().sort_values(ascending=False)
    
    # 3. Dynamické nastavenie výšky grafu
    plt_height = max(6, len(feature_names) * 0.3)
    
    plt.figure(figsize=(12, plt_height))
    
    # Vykreslenie všetkých príznakov
    sns.barplot(
        x=mean_imp.values, 
        y=mean_imp.index, 
        palette="viridis"  # Iná farebná schéma pre RF
    )
    
    plt.title(f"Kompletná globálna dôležitosť čŕt - Random Forest (Priemer cez {len(all_importances)} používateľov)", fontsize=16)
    plt.xlabel("Priemerný prínos (Feature Importance)", fontsize=12)
    plt.ylabel("Názov biometrického príznaku", fontsize=12)
    
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Uloženie s vysokým rozlíšením
    plt.savefig("global_feature_importance_RF_ALL.png", dpi=300)
    plt.show()

    # Vypísanie kompletného zoznamu do konzoly
    print("\n" + "="*50)
    print("KOMPLETNÝ REBRÍČEK DÔLEŽITOSTI - RANDOM FOREST (Zoradené):")
    print("="*50)
    for i, (name, val) in enumerate(mean_imp.items(), 1):
        print(f"{i:3d}. {name:30s} | Importance: {val:.6f}")
    print("="*50)

def plot_eer_histogram(df_results, max_acceptable_eer):
    """
    Histogram rozloženia EER hodnôt naprieč používateľmi.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df_results["EER"], bins=50, kde=True, color="forestgreen", edgecolor="black")
    
    # Pridáme čiaru pre priemerné EER
    avg_eer = df_results["EER"].mean()
    plt.axvline(avg_eer, color="red", linestyle="dashed", linewidth=2, label=f"Priemerné EER: {avg_eer:.4f}")
    
    # Pridáme čiaru pre prah akceptovateľnosti
    plt.axvline(max_acceptable_eer, color="orange", linestyle="dashed", linewidth=2, label=f"Max akceptovateľné: {max_acceptable_eer}")
    
    plt.title("Rozloženie Equal Error Rate (EER) - Random Forest", fontsize=14)
    plt.xlabel("EER (Hodnota)", fontsize=12)
    plt.ylabel("Počet používateľov", fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    
    # Uloženie do súboru
    plt.savefig("eer_histogram_RF.png", dpi=300)
    plt.show()
    plt.close()

def plot_sorted_eer(df_results):
    """
    Graf zoradených EER hodnôt od najlepšieho po najhorší model.
    """
    # Zoradíme dáta od najmenšieho EER
    sorted_eer = df_results["EER"].sort_values().values
    user_index = range(1, len(sorted_eer) + 1)

    plt.figure(figsize=(10, 6))
    plt.scatter(user_index, sorted_eer, color='forestgreen', s=15, label='EER používateľa')
    plt.fill_between(user_index, sorted_eer, color='forestgreen', alpha=0.1)

    plt.axhline(y=0.15, color='orange', linestyle='--', label='Prah 0.15')
    
    plt.title("EER zoradené podľa kvality modelu - Random Forest (1 = najlepší)")
    plt.xlabel("Poradie používateľa")
    plt.ylabel("EER")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("eer_sorted_RF.png", dpi=300)
    plt.show()
    plt.close()

# ============================================================================
# HLAVNÝ PROGRAM
# ============================================================================

# 1. Načítanie dát
df_train = pd.read_csv("train-ws22-st4.csv")
df_test = pd.read_csv("test-ws22-st4 copy.csv")

# 2. Odstránenie problematických používateľov (voliteľné)
users_to_drop = [4, 103]  # Uprav podľa potreby
df_train, df_test = remove_users(df_train, df_test, users_to_drop)

# 3. Definícia stĺpcov na odstránenie
columns_to_drop = ["user_id", "csv_file", "window_id"]
# Ak chceš odstrániť ďalšie príznaky, pridaj ich sem:
# columns_to_drop = ["user_id", "csv_file", "window_id", "std_angle_std", "std_angle_mean", ...]

# 4. Získanie zoznamu používateľov
all_users = sorted(df_train["user_id"].unique())

# 5. Nastavenia
MAX_ACCEPTABLE_EER = 0.15
global_results = []
all_importances = []

# Voliteľné: používatelia pre XAI analýzu
users_for_xai = []  # Napr. [all_users[0], all_users[1]]

print(f"Začínam trénovanie Random Forest modelov pre {len(all_users)} používateľov...\n")

# ============================================================================
# HLAVNÝ CYKLUS - TRÉNOVANIE PRE KAŽDÉHO POUŽÍVATEĽA
# ============================================================================

for target_user_id in all_users:
    # Vytvorenie kópií pre aktuálnu iteráciu
    df_train_user = df_train.copy()
    df_test_user = df_test.copy()
    
    # Vytvorenie binárnych labelov pre aktuálneho používateľa
    df_train_user["label"] = (df_train_user.user_id == target_user_id).astype(int)
    df_test_user["label"] = (df_test_user.user_id == target_user_id).astype(int)
    
    # Použijeme celý dataset (bez undersampligu, ako v XGBoost verzii)
    df_train_full = df_train_user.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Príprava matíc pre model
    x_train = df_train_full.drop(columns=columns_to_drop + ["label"], errors='ignore')
    y_train = df_train_full["label"]

    x_test = df_test_user.drop(columns=columns_to_drop + ["label"], errors='ignore')
    y_test = df_test_user["label"]
    
    # Výpočet pomerov tried pre class_weight
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    
    # Prevencia proti chybám: preskočíme používateľa, ak v teste alebo tréningu nemá žiadne pozitívne vzorky
    if y_test.sum() == 0 or pos_count == 0:
        print(f"Používateľ {target_user_id:3d} | PRESKOČENÝ (chýbajú vzorky)")
        continue

    # Výpočet scale_pos_weight (ekvivalent pre XGBoost)
    scale_pos_weight = neg_count / pos_count

    # ========================================================================
    # RANDOM FOREST MODEL - OPTIMALIZOVANÉ NASTAVENIE
    # ========================================================================
    rf = RandomForestClassifier(
        # === POČET STROMOV ===
        n_estimators=500,           # RF potrebuje viac stromov ako boosting,
                                    # pretože stromy sú nezávislé (nie sekvenčné).
                                    # 500 je dobrý štart, viac pomáha ale spomaľuje.

        # === ŠTRUKTÚRA STROMOV ===
        max_depth=8,                # VYŠŠIE ako pri XGBoost/LGBM. RF stromy sú
                                    # nezávislé, takže jednotlivý strom potrebuje
                                    # väčšiu kapacitu. Overfitting rieši bagging.

        min_samples_split=10,       # Minimálny počet vzoriek na rozdelenie uzla.
                                    # Pri nevyváženosti (1:119) je 10 bezpečné.

        min_samples_leaf=5,         # Minimálny počet vzoriek v liste.
                                    # Zabraňuje splitom na 1-2 vzorkách.

        # === STOCHASTICKÉ VZORKOVANIE ===
        max_features=0.6,           # Každý strom vidí 60% čŕt (ekvivalent
                                    # colsample_bytree). Kľúčová regularizácia v RF.

        max_samples=0.8,            # Každý strom sa trénuje na 80% vzoriek.
                                    # Bootstrap s obmedzením — extra regularizácia.

        bootstrap=True,             # Štandardný bagging — vzorkovanie s opakovaním.

        # === NEVYVÁŽENOSŤ ===
        class_weight={0: 1, 1: scale_pos_weight},
                                    # Ekvivalent scale_pos_weight v XGBoost.
                                    # Zvyšuje váhu pozitívnej triedy.

        # === OSTATNÉ ===
        criterion='gini',           # Gini impurity — štandard pre klasifikáciu.
        random_state=42,
        n_jobs=-1,
    )

    # Trénovanie modelu
    rf.fit(x_train, y_train)

    # Uloženie feature importance
    importances = rf.feature_importances_
    all_importances.append(importances)

    # ========================================================================
    # PREDIKCIA A VYHODNOTENIE
    # ========================================================================
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
    print(f"Používateľ {target_user_id:3d} | "
          f"pos:{int(pos_count)} neg:{int(neg_count)} "
          f"| AUC: {roc_auc:.4f} | EER: {EER:.6f}")
    
    # Voliteľná SHAP analýza pre vybraných používateľov
    if target_user_id in users_for_xai:
        plot_shap_summary(rf, x_test, target_user_id)

# ============================================================================
# GLOBÁLNE VYHODNOTENIE
# ============================================================================

df_results = pd.DataFrame(global_results)

avg_auc = df_results["AUC"].mean()
max_auc = df_results["AUC"].max()
min_auc = df_results["AUC"].min()

avg_eer = df_results["EER"].mean()
max_eer = df_results["EER"].max()
min_eer = df_results["EER"].min()

unacceptable_eer_count = (df_results["EER"] > MAX_ACCEPTABLE_EER).sum()
total_users_evaluated = len(df_results)

print("\n" + "="*50)
print(" GLOBÁLNE VYHODNOTENIE - RANDOM FOREST")
print("="*50)
print(f"Počet úspešne vyhodnotených používateľov: {total_users_evaluated}")
print(f"{'-'*50}")
print(f"AUC   avg: {avg_auc:.4f} | min: {min_auc:.4f} | max: {max_auc:.4f}")
print(f"EER   avg: {avg_eer:.4f} | min: {min_eer:.4f} | max: {max_eer:.4f}")
print(f"{'-'*50}")
print(f"Používatelia s neprípustným EER (> {MAX_ACCEPTABLE_EER:.2f}): "
      f"{unacceptable_eer_count} "
      f"({unacceptable_eer_count/total_users_evaluated:.1%})")
print("="*50)

# ============================================================================
# VIZUALIZÁCIE (odkomentuj podľa potreby)
# ============================================================================

# Globálna feature importance naprieč všetkými používateľmi
plot_global_feature_importance(all_importances, x_train.columns)

# Histogram EER hodnôt
plot_eer_histogram(df_results, MAX_ACCEPTABLE_EER)

# Zoradené EER hodnoty
plot_sorted_eer(df_results)