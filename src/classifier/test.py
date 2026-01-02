import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from collections import defaultdict
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. NAČÍTANIE DÁT
# ============================================================
train_df = pd.read_csv("ALL_USERS_FEATURES.csv")
test_df = pd.read_csv("TEST_FEATURES.csv")

feature_cols = [col for col in train_df.columns 
                if col not in ["user_id", "segment_id", "csv_file"]]

# ============================================================
# 2. AGREGÁCIA SEGMENTOV
# ============================================================
def aggregate_user_segments(df, feature_cols, segments_per_sample=5):
    """
    Zoskupí segmenty používateľa do väčších vzoriek
    """
    aggregated_data = []
    
    for user_id in df['user_id'].unique():
        user_segments = df[df['user_id'] == user_id][feature_cols].values
        
        n_segments = len(user_segments)
        for i in range(0, n_segments, segments_per_sample):
            segment_group = user_segments[i:i+segments_per_sample]
            
            if len(segment_group) >= segments_per_sample // 2:
                # Agreguj features: mean, std, min, max
                features = {
                    'user_id': user_id,
                    **{f'{col}_mean': segment_group[:, j].mean() 
                       for j, col in enumerate(feature_cols)},
                    **{f'{col}_std': segment_group[:, j].std() 
                       for j, col in enumerate(feature_cols)},
                    **{f'{col}_min': segment_group[:, j].min() 
                       for j, col in enumerate(feature_cols)},
                    **{f'{col}_max': segment_group[:, j].max() 
                       for j, col in enumerate(feature_cols)},
                }
                aggregated_data.append(features)
    
    return pd.DataFrame(aggregated_data)

print("Agregácia segmentov...")
train_agg = aggregate_user_segments(train_df, feature_cols, segments_per_sample=5)
test_agg = aggregate_user_segments(test_df, feature_cols, segments_per_sample=5)

# ============================================================
# 3. AUTENTIFIKÁCIA: ONE-VS-REST PRÍSTUP
# ============================================================

class UserAuthenticator:
    """
    Autentifikátor - pre každého používateľa samostatný model
    """
    def __init__(self):
        self.models = {}
        self.scaler = None  # JEDNA spoločná normalizácia!
        self.users = []
        self.feature_columns = None
        
    def train(self, train_df):
        """
        Trénuje binárny klasifikátor pre každého používateľa
        """
        self.users = train_df['user_id'].unique()
        X = train_df.drop(columns=['user_id'])
        self.feature_columns = X.columns.tolist()
        
        # JEDNA spoločná normalizácia pre všetkých
        print("Normalizácia dát...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        for user_id in self.users:
            print(f"Trénovanie modelu pre používateľa: {user_id}")
            
            # Vytvor binárne labels: 1 = tento používateľ, 0 = všetci ostatní
            y_binary = (train_df['user_id'] == user_id).astype(int)
            
            # Tréning Random Forest s agresívnejšími parametrami
            model = RandomForestClassifier(
                n_estimators=500,           # Viac stromov
                max_depth=30,               # Hlbšie stromy
                min_samples_split=2,        # Citlivejšie delenie
                min_samples_leaf=1,         # Citlivejšie listy
                max_features='sqrt',
                class_weight={0: 1, 1: 10}, # Viac váhy pre pozitívnu triedu!
                n_jobs=-1,
                random_state=42
            )
            
            try:
                model.fit(X_scaled, y_binary)
                self.models[user_id] = model
            except Exception as e:
                print(f"  ⚠️ Chyba pri trénovaní {user_id}: {e}")
                continue
        
        print(f"\n✓ Natrénovaných {len(self.models)} modelov")
    
    def authenticate(self, X_test, claimed_user_id, threshold=0.5):
        """
        Autentifikácia: Je to naozaj claimed_user_id?
        
        Returns:
            - predictions: 1 = autentifikovaný, 0 = zamietnutý
            - probabilities: pravdepodobnosť autenticity
        """
        if claimed_user_id not in self.models:
            return np.array([0]), np.array([0.0])
        
        model = self.models[claimed_user_id]
        
        # Zabezpeč správne stĺpce
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test[self.feature_columns]
            X_array = X_test.values
        else:
            X_array = X_test
        
        # Normalizuj
        X_scaled = self.scaler.transform(X_array)
        
        # Predikuj
        try:
            # Skús predict_proba
            proba = model.predict_proba(X_scaled)
            
            if proba.shape[1] == 2:
                probabilities = proba[:, 1]
            elif len(model.classes_) == 1:
                # Model videl len jednu triedu
                if model.classes_[0] == 1:
                    probabilities = np.ones(len(X_scaled))
                else:
                    probabilities = np.zeros(len(X_scaled))
            else:
                probabilities = proba[:, 0] if model.classes_[0] == 1 else 1 - proba[:, 0]
                
        except Exception as e:
            # Ak zlyhá predict_proba, použi predict
            print(f"  Fallback na predict() pre {claimed_user_id}")
            predictions = model.predict(X_scaled)
            probabilities = predictions.astype(float)
            return predictions, probabilities
        
        predictions = (probabilities >= threshold).astype(int)
        return predictions, probabilities
    
    def save(self, filename='authenticator.pkl'):
        """Ulož natrénovaný systém"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filename='authenticator.pkl'):
        """Načítaj natrénovaný systém"""
        with open(filename, 'rb') as f:
            return pickle.load(f)

# ============================================================
# 4. TRÉNOVANIE
# ============================================================
authenticator = UserAuthenticator()
authenticator.train(train_agg)

# ============================================================
# 5. TESTOVANIE AUTENTIFIKÁCIE
# ============================================================
print("\n" + "="*60)
print("TESTOVANIE AUTENTIFIKÁCIE")
print("="*60)

THRESHOLD = 0.2  # ZNÍŽENÝ PRAH! (bolo 0.5)

X_test = test_agg.drop(columns=['user_id'])
y_test = test_agg['user_id']

# Vyhodnotenie pre každého používateľa
results = defaultdict(lambda: {
    'true_positives': 0,
    'false_positives': 0,
    'true_negatives': 0,
    'false_negatives': 0,
    'probs': []
})

print(f"Testovanie {len(X_test)} vzoriek...")

# Pre každú testovaciu vzorku
for idx, (_, row) in enumerate(X_test.iterrows()):
    if idx % 100 == 0:
        print(f"  Spracovaných: {idx}/{len(X_test)}")
    
    X_sample = row.values.reshape(1, -1)
    true_user = y_test.iloc[idx]
    
    # Test 1: Autentifikácia ako správny používateľ
    if true_user in authenticator.models:
        pred, prob = authenticator.authenticate(X_sample, true_user, threshold=THRESHOLD)
        
        if pred[0] == 1:
            results[true_user]['true_positives'] += 1
        else:
            results[true_user]['false_negatives'] += 1
        
        results[true_user]['probs'].append(prob[0])
    
    # Test 2: Pokus o autentifikáciu ako iný používateľ (impostor test)
    # Testuj len 5 náhodných používateľov pre rýchlosť
    other_users = [u for u in authenticator.users if u != true_user]
    test_users = np.random.choice(other_users, min(5, len(other_users)), replace=False)
    
    for other_user in test_users:
        pred, prob = authenticator.authenticate(X_sample, other_user, threshold=THRESHOLD)
        
        if pred[0] == 1:
            results[other_user]['false_positives'] += 1
        else:
            results[other_user]['true_negatives'] += 1

# ============================================================
# 6. VÝPOČET METRÍK
# ============================================================
print("\n" + "="*60)
print("VÝSLEDKY PRE KAŽDÉHO POUŽÍVATEĽA")
print("="*60)

summary_stats = []

for user_id in sorted(authenticator.users):
    tp = results[user_id]['true_positives']
    fp = results[user_id]['false_positives']
    tn = results[user_id]['true_negatives']
    fn = results[user_id]['false_negatives']
    
    # Metriky
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    summary_stats.append({
        'user_id': user_id,
        'FAR': far,
        'FRR': frr,
        'Accuracy': accuracy,
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn
    })
    
    print(f"\nPoužívateľ: {user_id}")
    print(f"  FAR (False Accept Rate):  {far:.4f} ({far*100:.2f}%) ← impostorov prijatých")
    print(f"  FRR (False Reject Rate):  {frr:.4f} ({frr*100:.2f}%) ← legit odmietnutých")
    print(f"  Accuracy:                 {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Samples: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

print("\n" + "="*60)
print("ANALÝZA PROBLÉMOV")
print("="*60)

# Ukáž distribúciu pravdepodobností
all_probs = []
for user_id in authenticator.users:
    all_probs.extend(results[user_id]['probs'])

if all_probs:
    all_probs = np.array(all_probs)
    print(f"Pravdepodobnosti pre legitímnych používateľov:")
    print(f"  Min:     {all_probs.min():.4f}")
    print(f"  Mean:    {all_probs.mean():.4f}")
    print(f"  Median:  {np.median(all_probs):.4f}")
    print(f"  Max:     {all_probs.max():.4f}")
    print(f"  Std:     {all_probs.std():.4f}")
    print(f"\nOdporúčaný prah: {np.percentile(all_probs, 20):.4f} (20. percentil)")

# Priemerné metriky
summary_df = pd.DataFrame(summary_stats)
print("\n" + "="*60)
print("PRIEMERNÉ METRIKY SYSTÉMU")
print("="*60)
print(f"Priemerná FAR: {summary_df['FAR'].mean():.4f} ({summary_df['FAR'].mean()*100:.2f}%)")
print(f"Priemerná FRR: {summary_df['FRR'].mean():.4f} ({summary_df['FRR'].mean()*100:.2f}%)")
print(f"Priemerná Accuracy: {summary_df['Accuracy'].mean():.4f} ({summary_df['Accuracy'].mean()*100:.2f}%)")

# ============================================================
# 7. ULOŽENIE MODELU
# ============================================================
authenticator.save('user_authenticator.pkl')
print("\n✓ Autentifikátor uložený ako 'user_authenticator.pkl'")

# ============================================================
# 8. PRÍKLAD POUŽITIA
# ============================================================
print("\n" + "="*60)
print("PRÍKLAD AUTENTIFIKÁCIE")
print("="*60)

auth = UserAuthenticator.load('user_authenticator.pkl')

sample = X_test.iloc[0:1].values
claimed_id = y_test.iloc[0]

pred, prob = auth.authenticate(sample, claimed_id, threshold=0.5)

print(f"Deklarovaný používateľ: {claimed_id}")
print(f"Pravdepodobnosť autenticity: {prob[0]:.4f}")
print(f"Rozhodnutie: {'✓ AUTENTIFIKOVANÝ' if pred[0] == 1 else '✗ ZAMIETNUTÝ'}")

print("\nVplyv prahu:")
for thr in [0.3, 0.5, 0.7, 0.9]:
    pred, prob = auth.authenticate(sample, claimed_id, threshold=thr)
    print(f"  Prah {thr:.1f}: {'✓ OK' if pred[0] == 1 else '✗ NO'} (prob={prob[0]:.4f})")