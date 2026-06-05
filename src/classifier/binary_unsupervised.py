import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
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

def plot_eer_comparison(df_results_if, df_results_ocsvm, max_acceptable_eer):
    """
    Porovnanie EER histogramov pre oba modely vedľa seba.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Isolation Forest
    ax1 = axes[0]
    sns.histplot(df_results_if["EER"], bins=50, kde=True, color="purple", edgecolor="black", ax=ax1)
    avg_eer_if = df_results_if["EER"].mean()
    ax1.axvline(avg_eer_if, color="red", linestyle="dashed", linewidth=2, label=f"Priemerné EER: {avg_eer_if:.4f}")
    ax1.axvline(max_acceptable_eer, color="orange", linestyle="dashed", linewidth=2, label=f"Max akceptovateľné: {max_acceptable_eer}")
    ax1.set_title("Isolation Forest - EER Distribution", fontsize=14, fontweight='bold')
    ax1.set_xlabel("EER (Hodnota)", fontsize=12)
    ax1.set_ylabel("Počet používateľov", fontsize=12)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.75)
    
    # One-Class SVM
    ax2 = axes[1]
    sns.histplot(df_results_ocsvm["EER"], bins=50, kde=True, color="darkblue", edgecolor="black", ax=ax2)
    avg_eer_ocsvm = df_results_ocsvm["EER"].mean()
    ax2.axvline(avg_eer_ocsvm, color="red", linestyle="dashed", linewidth=2, label=f"Priemerné EER: {avg_eer_ocsvm:.4f}")
    ax2.axvline(max_acceptable_eer, color="orange", linestyle="dashed", linewidth=2, label=f"Max akceptovateľné: {max_acceptable_eer}")
    ax2.set_title("One-Class SVM - EER Distribution", fontsize=14, fontweight='bold')
    ax2.set_xlabel("EER (Hodnota)", fontsize=12)
    ax2.set_ylabel("Počet používateľov", fontsize=12)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.75)
    
    plt.tight_layout()
    plt.savefig("eer_comparison_unsupervised.png", dpi=300)
    plt.show()
    plt.close()

def plot_sorted_eer_comparison(df_results_if, df_results_ocsvm):
    """
    Porovnanie zoradených EER hodnôt pre oba modely.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Isolation Forest
    sorted_eer_if = df_results_if["EER"].sort_values().values
    user_index_if = range(1, len(sorted_eer_if) + 1)
    
    ax1 = axes[0]
    ax1.scatter(user_index_if, sorted_eer_if, color='purple', s=15, label='EER používateľa')
    ax1.fill_between(user_index_if, sorted_eer_if, color='purple', alpha=0.1)
    ax1.axhline(y=0.15, color='orange', linestyle='--', label='Prah 0.15')
    ax1.set_title("Isolation Forest - Zoradené EER", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Poradie používateľa")
    ax1.set_ylabel("EER")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # One-Class SVM
    sorted_eer_ocsvm = df_results_ocsvm["EER"].sort_values().values
    user_index_ocsvm = range(1, len(sorted_eer_ocsvm) + 1)
    
    ax2 = axes[1]
    ax2.scatter(user_index_ocsvm, sorted_eer_ocsvm, color='darkblue', s=15, label='EER používateľa')
    ax2.fill_between(user_index_ocsvm, sorted_eer_ocsvm, color='darkblue', alpha=0.1)
    ax2.axhline(y=0.15, color='orange', linestyle='--', label='Prah 0.15')
    ax2.set_title("One-Class SVM - Zoradené EER", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Poradie používateľa")
    ax2.set_ylabel("EER")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("eer_sorted_comparison_unsupervised.png", dpi=300)
    plt.show()
    plt.close()

def plot_model_comparison_boxplot(df_results_if, df_results_ocsvm):
    """
    Boxplot porovnanie výkonnosti oboch modelov.
    """
    # Príprava dát pre boxplot
    data_for_plot = pd.DataFrame({
        'Model': ['Isolation Forest'] * len(df_results_if) + ['One-Class SVM'] * len(df_results_ocsvm),
        'EER': list(df_results_if['EER']) + list(df_results_ocsvm['EER']),
        'AUC': list(df_results_if['AUC']) + list(df_results_ocsvm['AUC'])
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # EER Boxplot
    sns.boxplot(data=data_for_plot, x='Model', y='EER', ax=axes[0], palette=['purple', 'darkblue'])
    axes[0].set_title('Porovnanie EER', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('EER', fontsize=12)
    axes[0].axhline(y=0.15, color='orange', linestyle='--', linewidth=2, label='Prah 0.15')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # AUC Boxplot
    sns.boxplot(data=data_for_plot, x='Model', y='AUC', ax=axes[1], palette=['purple', 'darkblue'])
    axes[1].set_title('Porovnanie AUC', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('AUC', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("model_comparison_boxplot.png", dpi=300)
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

# 4. Získanie zoznamu používateľov
all_users = sorted(df_train["user_id"].unique())

# 5. Nastavenia
MAX_ACCEPTABLE_EER = 0.15

# Výsledky pre oba modely
results_isolation_forest = []
results_ocsvm = []

print(f"{'='*70}")
print(f"UNSUPERVISED LEARNING - ANOMALY DETECTION")
print(f"Trénovanie len na pozitívnych vzorkách (genuine samples)")
print(f"{'='*70}\n")

print(f"Začínam trénovanie pre {len(all_users)} používateľov...\n")
print(f"{'User ID':>8} | {'Pos':>5} | {'Neg':>5} | {'IF_AUC':>7} | {'IF_EER':>8} | {'SVM_AUC':>7} | {'SVM_EER':>8}")
print(f"{'-'*80}")

# ============================================================================
# HLAVNÝ CYKLUS - TRÉNOVANIE PRE KAŽDÉHO POUŽÍVATEĽA
# ============================================================================

for target_user_id in all_users:
    # Vytvorenie kópií pre aktuálnu iteráciu
    df_train_user = df_train.copy()
    df_test_user = df_test.copy()
    
    # Vytvorenie binárnych labelov
    df_train_user["label"] = (df_train_user.user_id == target_user_id).astype(int)
    df_test_user["label"] = (df_test_user.user_id == target_user_id).astype(int)
    
    # ========================================================================
    # KĽÚČOVÝ ROZDIEL: Trénovanie LEN NA POZITÍVNYCH VZORKÁCH
    # ========================================================================
    df_train_genuine = df_train_user[df_train_user["label"] == 1].copy()
    
    # Príprava trénovacích dát (LEN genuine samples)
    x_train_genuine = df_train_genuine.drop(columns=columns_to_drop + ["label"], errors='ignore')
    
    # Test set obsahuje BOTH genuine a impostor samples
    x_test = df_test_user.drop(columns=columns_to_drop + ["label"], errors='ignore')
    y_test = df_test_user["label"]
    
    # Výpočet pomerov tried
    pos_count = len(df_train_genuine)
    neg_count = len(df_train_user) - pos_count
    
    # Prevencia proti chybám
    if y_test.sum() == 0 or pos_count == 0:
        print(f"{target_user_id:8d} | PRESKOČENÝ (chýbajú vzorky)")
        continue
    
    # ========================================================================
    # NORMALIZÁCIA DÁT (dôležité pre One-Class SVM)
    # ========================================================================
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_genuine)
    x_test_scaled = scaler.transform(x_test)
    
    # ========================================================================
    # MODEL 1: ISOLATION FOREST
    # ========================================================================
    # Isolation Forest deteguje anomálie pomocou náhodných stromov
    # Trénuje sa LEN na genuine samples, impostor samples budú anomálie
    
    iso_forest = IsolationForest(
        n_estimators=200,           # Počet stromov
        max_samples='auto',         # Počet vzoriek na strom (automaticky min(256, n_samples))
        contamination=0.1,          # Očakávaný podiel anomálií (10% je konzervatívne)
        max_features=1.0,           # Počet čŕt na split (1.0 = všetky)
        bootstrap=False,            # Nepoužívať bootstrap sampling
        random_state=42,
        n_jobs=-1
    )
    
    # Trénovanie len na genuine samples
    iso_forest.fit(x_train_genuine)
    
    # Predikcia: decision_function vracia anomaly score
    # Čím NIŽŠIA hodnota, tým väčšia pravdepodobnosť anomálie
    # Pre ROC krivku potrebujeme VYŠŠIE skóre = genuine, takže invertujeme
    scores_if = -iso_forest.decision_function(x_test)
    
    # Výpočet metrík
    fpr_if, tpr_if, _ = roc_curve(y_test, scores_if)
    auc_if = auc(fpr_if, tpr_if)
    
    fnr_if = 1 - tpr_if
    eer_index_if = np.nanargmin(np.abs(fnr_if - fpr_if))
    eer_if = fpr_if[eer_index_if]
    
    # ========================================================================
    # MODEL 2: ONE-CLASS SVM
    # ========================================================================
    # One-Class SVM sa snaží nájsť hranicu okolo genuine samples
    # Všetko mimo túto hranicu je považované za anomáliu
    
    ocsvm = OneClassSVM(
        kernel='rbf',               # Radiálna bázová funkcia (najčastejšie používané)
        gamma='scale',              # Automatické nastavenie gamma (1 / (n_features * X.var()))
        nu=0.1,                     # Horná hranica na podiel outlierov (0.1 = 10%)
                                    # Nu je ekvivalent contamination v Isolation Forest
        shrinking=True,             # Použiť shrinking heuristic pre rýchlejší tréning
        cache_size=200,             # Veľkosť cache v MB
    )
    
    # Trénovanie len na genuine samples (SCALED!)
    ocsvm.fit(x_train_scaled)
    
    # Predikcia: decision_function vracia vzdialenosť od hranice
    # Čím VYŠŠIA hodnota, tým viac "inside" genuine distribúcie
    scores_ocsvm = ocsvm.decision_function(x_test_scaled)
    
    # Výpočet metrík
    fpr_ocsvm, tpr_ocsvm, _ = roc_curve(y_test, scores_ocsvm)
    auc_ocsvm = auc(fpr_ocsvm, tpr_ocsvm)
    
    fnr_ocsvm = 1 - tpr_ocsvm
    eer_index_ocsvm = np.nanargmin(np.abs(fnr_ocsvm - fpr_ocsvm))
    eer_ocsvm = fpr_ocsvm[eer_index_ocsvm]
    
    # ========================================================================
    # ULOŽENIE VÝSLEDKOV
    # ========================================================================
    results_isolation_forest.append({
        "user_id": target_user_id,
        "AUC": auc_if,
        "EER": eer_if
    })
    
    results_ocsvm.append({
        "user_id": target_user_id,
        "AUC": auc_ocsvm,
        "EER": eer_ocsvm
    })
    
    # Priebežný výpis
    print(f"{target_user_id:8d} | {pos_count:5d} | {neg_count:5d} | "
          f"{auc_if:7.4f} | {eer_if:8.6f} | "
          f"{auc_ocsvm:7.4f} | {eer_ocsvm:8.6f}")

# ============================================================================
# GLOBÁLNE VYHODNOTENIE
# ============================================================================

df_results_if = pd.DataFrame(results_isolation_forest)
df_results_ocsvm = pd.DataFrame(results_ocsvm)

print(f"\n{'='*70}")
print(f" GLOBÁLNE VYHODNOTENIE - ISOLATION FOREST")
print(f"{'='*70}")

avg_auc_if = df_results_if["AUC"].mean()
avg_eer_if = df_results_if["EER"].mean()
min_eer_if = df_results_if["EER"].min()
max_eer_if = df_results_if["EER"].max()
unacceptable_if = (df_results_if["EER"] > MAX_ACCEPTABLE_EER).sum()

print(f"Počet vyhodnotených používateľov: {len(df_results_if)}")
print(f"{'-'*70}")
print(f"AUC   avg: {avg_auc_if:.4f} | min: {df_results_if['AUC'].min():.4f} | max: {df_results_if['AUC'].max():.4f}")
print(f"EER   avg: {avg_eer_if:.4f} | min: {min_eer_if:.4f} | max: {max_eer_if:.4f}")
print(f"{'-'*70}")
print(f"Používatelia s neprípustným EER (> {MAX_ACCEPTABLE_EER:.2f}): "
      f"{unacceptable_if} ({unacceptable_if/len(df_results_if):.1%})")

print(f"\n{'='*70}")
print(f" GLOBÁLNE VYHODNOTENIE - ONE-CLASS SVM")
print(f"{'='*70}")

avg_auc_ocsvm = df_results_ocsvm["AUC"].mean()
avg_eer_ocsvm = df_results_ocsvm["EER"].mean()
min_eer_ocsvm = df_results_ocsvm["EER"].min()
max_eer_ocsvm = df_results_ocsvm["EER"].max()
unacceptable_ocsvm = (df_results_ocsvm["EER"] > MAX_ACCEPTABLE_EER).sum()

print(f"Počet vyhodnotených používateľov: {len(df_results_ocsvm)}")
print(f"{'-'*70}")
print(f"AUC   avg: {avg_auc_ocsvm:.4f} | min: {df_results_ocsvm['AUC'].min():.4f} | max: {df_results_ocsvm['AUC'].max():.4f}")
print(f"EER   avg: {avg_eer_ocsvm:.4f} | min: {min_eer_ocsvm:.4f} | max: {max_eer_ocsvm:.4f}")
print(f"{'-'*70}")
print(f"Používatelia s neprípustným EER (> {MAX_ACCEPTABLE_EER:.2f}): "
      f"{unacceptable_ocsvm} ({unacceptable_ocsvm/len(df_results_ocsvm):.1%})")

# ============================================================================
# POROVNANIE MODELOV
# ============================================================================

print(f"\n{'='*70}")
print(f" POROVNANIE MODELOV")
print(f"{'='*70}")
print(f"{'Metrika':<30} | {'Isolation Forest':>15} | {'One-Class SVM':>15}")
print(f"{'-'*70}")
print(f"{'Priemerné AUC':<30} | {avg_auc_if:>15.4f} | {avg_auc_ocsvm:>15.4f}")
print(f"{'Priemerné EER':<30} | {avg_eer_if:>15.4f} | {avg_eer_ocsvm:>15.4f}")
print(f"{'Min EER':<30} | {min_eer_if:>15.4f} | {min_eer_ocsvm:>15.4f}")
print(f"{'Max EER':<30} | {max_eer_if:>15.4f} | {max_eer_ocsvm:>15.4f}")
print(f"{'Neprípustné EER (count)':<30} | {unacceptable_if:>15d} | {unacceptable_ocsvm:>15d}")
print(f"{'Neprípustné EER (%)':<30} | {unacceptable_if/len(df_results_if):>14.1%} | {unacceptable_ocsvm/len(df_results_ocsvm):>14.1%}")
print(f"{'='*70}")

# Určenie lepšieho modelu
if avg_eer_if < avg_eer_ocsvm:
    winner = "Isolation Forest"
    diff = avg_eer_ocsvm - avg_eer_if
else:
    winner = "One-Class SVM"
    diff = avg_eer_if - avg_eer_ocsvm

print(f"\n🏆 VÍŤAZ: {winner} (lepší o {diff:.4f} EER)")

# ============================================================================
# VIZUALIZÁCIE
# ============================================================================

print("\nGenerujem vizualizácie...")

# Porovnanie EER histogramov
plot_eer_comparison(df_results_if, df_results_ocsvm, MAX_ACCEPTABLE_EER)

# Porovnanie zoradených EER
plot_sorted_eer_comparison(df_results_if, df_results_ocsvm)

# Boxplot porovnanie
plot_model_comparison_boxplot(df_results_if, df_results_ocsvm)

print("\n✓ Všetky vizualizácie boli vytvorené a uložené!")

# ============================================================================
# EXPORT VÝSLEDKOV DO CSV
# ============================================================================

# Spojenie výsledkov pre export
df_results_combined = pd.merge(
    df_results_if,
    df_results_ocsvm,
    on='user_id',
    suffixes=('_IF', '_OCSVM')
)

df_results_combined.to_csv("unsupervised_results_comparison.csv", index=False)
print("\n✓ Výsledky exportované do: unsupervised_results_comparison.csv")