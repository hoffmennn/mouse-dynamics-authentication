import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

# --- 1. POMOCNÉ FUNKCIE ---

def get_feature_columns(df):
    """
    Vyberie stĺpce pre trénovanie.
    POZNÁMKA: Ak chcete zvýšiť presnosť (AUC), odporúčam zmazať tie 
    vylúčené 'click' atribúty, ak ich dáta obsahujú.
    """
    exclude_cols = [
        "user_id", "csv_file", "session_id", 
        # Tieto atribúty ste pôvodne vyhadzovali - ak ich chcete vrátiť, zmažte ich z tohto zoznamu:
        #"num_clicks", "mean_click_duration", "std_click_duration",
        #"mean_inter_click_time", "std_inter_click_time"
    ]
    return [c for c in df.columns if c not in exclude_cols]

def compute_threshold_at_far(y_true, y_scores, target_far):
    """
    Vypočíta prahovú hodnotu (threshold), pri ktorej je FAR <= target_far.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Nájdeme indexy, kde je False Positive Rate pod naším cieľom
    eligible_indices = np.where(fpr <= target_far)[0]
    
    if len(eligible_indices) == 0:
        return thresholds[0] # Fallback (ak sa nedá dosiahnuť, berieme najprísnejší)
    
    # Berieme posledný index (najvyšší možný threshold, ktorý ešte spĺňa podmienku)
    return thresholds[eligible_indices[-1]]

# --- 2. HLAVNÁ LOGIKA (TRÉNING A VYHODNOTENIE) ---

def process_user_pipeline(user_id, train_df, test_df, target_far):
    features = get_feature_columns(train_df)
    
    # --- A. PRÍPRAVA DÁT PRE TRÉNING ---
    # Vyberieme pozitívne vzorky (genuine user)
    pos_train_all = train_df[train_df.user_id == user_id]
    
    # Vyberieme negatívne vzorky (impostors) - Sampling
    neg_train_all = train_df[train_df.user_id != user_id]
    # Pomer 1:3 (Genuine:Impostor) pre tréning je zvyčajne postačujúci
    neg_train_sampled = neg_train_all.sample(n=min(len(neg_train_all), len(pos_train_all) * 3), random_state=42)
    
    # Spojíme do jedného datasetu
    X_full = pd.concat([pos_train_all, neg_train_sampled])[features]
    y_full = np.concatenate([np.ones(len(pos_train_all)), np.zeros(len(neg_train_sampled))])
    
    # --- B. ROZDELENIE NA TRAIN A VALIDATION (KRITICKÝ KROK) ---
    # Train: na tomto sa Random Forest učí
    # Val: na tomto hľadáme Threshold (aby sme predišli preučeniu/Data Leakage)
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.25, stratify=y_full, random_state=42
    )
    
    # --- C. TRÉNING MODELU ---
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1  # Využije všetky procesory
    )
    model.fit(X_train, y_train)
    
    # --- D. NÁJDENIE OPTIMÁLNEHO THRESHOLDU (na Validačnej množine) ---
    val_probs = model.predict_proba(X_val)[:, 1]
    optimal_threshold = compute_threshold_at_far(y_val, val_probs, target_far)
    
    # --- E. TESTOVANIE (na Testovacej množine - úplne nové dáta) ---
    pos_test = test_df[test_df.user_id == user_id]
    neg_test = test_df[test_df.user_id != user_id]
    
    # Sampling pre test (aby sme nemali milióny riadkov, napr. 1:10)
    neg_test_sampled = neg_test.sample(n=min(len(neg_test), len(pos_test) * 10), random_state=42)
    
    X_test = pd.concat([pos_test, neg_test_sampled])[features]
    # y_test: 1 pre genuine, 0 pre impostor
    y_test = np.concatenate([np.ones(len(pos_test)), np.zeros(len(neg_test_sampled))])
    
    # Predikcia pravdepodobností
    test_probs = model.predict_proba(X_test)[:, 1]
    
    # --- F. VÝPOČET METRÍK ---
    
    # 1. AUC (Area Under Curve) - nezávisí od thresholdu
    auc = roc_auc_score(y_test, test_probs)
    
    # 2. Aplikácia thresholdu (Binárne rozhodnutie)
    # Ak je pravdepodobnosť >= threshold -> Prijatý (1), inak Zamietnutý (0)
    predictions = (test_probs >= optimal_threshold).astype(int)
    
    # 3. FAR (False Acceptance Rate): Impostori (0), ktorých sme prijali (1)
    # Pozeráme sa len na riadky, kde y_test == 0
    neg_indices = (y_test == 0)
    far = np.mean(predictions[neg_indices] == 1)
    
    # 4. FRR (False Rejection Rate): Genuine (1), ktorých sme zamietli (0)
    # Pozeráme sa len na riadky, kde y_test == 1
    pos_indices = (y_test == 1)
    frr = np.mean(predictions[pos_indices] == 0)
    
    return {
        "user_id": user_id,
        "AUC": auc,
        "FAR": far,
        "FRR": frr,
        "Threshold": optimal_threshold
    }

# --- 3. SPUSTENIE ---

# Načítanie dát (zmeňte cesty k súborom podľa potreby)
try:
    print("Načítavam dáta...")
    train_df = pd.read_csv("ALL_USERS_FEATURES.csv")
    test_df  = pd.read_csv("TEST_FEATURES.csv")
except FileNotFoundError:
    print("Chyba: CSV súbory sa nenašli. Uistite sa, že sú v rovnakom priečinku.")
    # Vytvoríme dummy dáta pre ukážku, ak súbory neexistujú (pre testovanie kódu)
    # (Túto časť v reále zmažte)
    exit()

users = sorted(train_df.user_id.unique())
results = []

print(f"Spracovávam {len(users)} používateľov...")

for uid in users:
    # Pre každý model chceme nastaviť cieľové FAR na 2.5% (0.025)
    res = process_user_pipeline(uid, train_df, test_df, target_far=0.05)
    results.append(res)
    
    # Voliteľný print pre sledovanie postupu
    print(f"User {uid}: AUC={res['AUC']:.4f}, FAR={res['FAR']:.4f}, FRR={res['FRR']:.4f}")

# --- 4. VÝSLEDNÁ TABUĽKA ---

results_df = pd.DataFrame(results)

print("\n--- SÚHRNNÉ VÝSLEDKY ---")
print(results_df.describe())

# Uloženie výsledkov
# results_df.to_csv("final_results.csv", index=False)

print("\n--- PROBLÉMOVÍ POUŽÍVATELIA (AUC < 0.6) ---")
low_auc = results_df[results_df.AUC < 0.6]
if not low_auc.empty:
    print(low_auc[["user_id", "AUC"]])
else:
    print("Všetci používatelia majú AUC nad 0.6")