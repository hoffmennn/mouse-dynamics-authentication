import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

def compute_far_threshold(y_true, y_score, target_far=0.01):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.where(fpr <= target_far)[0]
    if len(idx) == 0:
        return thresholds[-1]
    return thresholds[idx[-1]]


def get_feature_columns(df):
    return [c for c in df.columns if c not in [
        "user_id", "csv_file",
        "num_clicks", "mean_click_duration", "std_click_duration",
        "mean_inter_click_time", "std_inter_click_time"
    ]]


def train_user_model(user_id, train_df, neg_ratio=3, target_far=0.01):

    pos = train_df[train_df.user_id == user_id]
    neg = train_df[train_df.user_id != user_id]

    neg_sampled = neg.sample(
        n=min(len(neg), len(pos) * neg_ratio),
        random_state=42
    )

    train_data = pd.concat([pos, neg_sampled])
    X = train_data[get_feature_columns(train_df)]
    y = (train_data.user_id == user_id).astype(int)

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]
    threshold = compute_far_threshold(y, probs, target_far)

    return model, threshold


def evaluate_user_model(
    user_id,
    model,
    threshold,
    test_df,
    neg_test_ratio=5
):
    pos = test_df[test_df.user_id == user_id]
    neg = test_df[test_df.user_id != user_id]

    neg_sampled = neg.sample(
        n=min(len(neg), len(pos) * neg_test_ratio),
        random_state=42
    )

    eval_data = pd.concat([pos, neg_sampled])
    X = eval_data[get_feature_columns(test_df)]
    y = (eval_data.user_id == user_id).astype(int)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    auc = roc_auc_score(y, probs)

    FAR = np.mean((y == 0) & (preds == 1))
    FRR = np.mean((y == 1) & (preds == 0))

    return auc, FAR, FRR


train_df = pd.read_csv("ALL_USERS_FEATURES.csv")
test_df  = pd.read_csv("TEST_FEATURES.csv")

users = sorted(train_df.user_id.unique())

results = []

for uid in users:
    print(f"=== USER {uid} ===")

    model, thr = train_user_model(uid, train_df)

    auc, far, frr = evaluate_user_model(
        uid,
        model,
        thr,
        test_df,
        neg_test_ratio=5   # kompromis
    )

    results.append({
        "user_id": uid,
        "AUC": auc,
        "FAR": far,
        "FRR": frr
    })

results_df = pd.DataFrame(results)


print("\n=== OVERALL RESULTS ===")
print(results_df.describe())

print("\nUsers with AUC < 0.6:")
print(results_df[results_df.AUC < 0.6][["user_id", "AUC"]])
