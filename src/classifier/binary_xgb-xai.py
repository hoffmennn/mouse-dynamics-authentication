import os
import pandas as pd
import numpy as np

import shap
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
from scipy.stats import spearmanr


# -- KONFIGURACIA --

TRAIN_CSV          = "train-ws22-st4.csv"
TEST_CSV           = "test-ws22-st4.csv"
USERS_TO_DROP      = [4, 103]
USERS_FOR_XAI      = {15, 80, 90}
MAX_ACCEPTABLE_EER = 0.15
COLUMNS_TO_DROP    = [
    "user_id", "csv_file", "window_id",
    "std_angle_std", "std_angle_mean",
    "scattering_coefficient_norm_std", "scattering_coefficient_norm_mean",
    "tcm_norm_std", "std_dt_std",
]

XGB_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.025,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.75,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.5,
    "reg_lambda": 3.0,
    "gamma": 0.3,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 15,
    "n_jobs": -1,
    "tree_method": "hist",
}


# -- POMOCNE FUNKCIE --

def remove_users(df_train, df_test, users_to_remove):
    before_train = len(df_train)
    before_test = len(df_test)

    df_train_cleaned = df_train[~df_train["user_id"].isin(users_to_remove)].copy()
    df_test_cleaned = df_test[~df_test["user_id"].isin(users_to_remove)].copy()

    print(f"--- CISTENIE DAT ---")
    print(f"Odstranujem pouzivatelov: {users_to_remove}")
    print(f"Train: odstranenych {before_train - len(df_train_cleaned)} riadkov.")
    print(f"Test:  odstranenych {before_test - len(df_test_cleaned)} riadkov.")
    print(f"--------------------\n")

    return df_train_cleaned, df_test_cleaned


def build_model(scale_pos_weight):
    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**params)


def compute_shap(model, x_data):
    """Vypocita SHAP hodnoty s opravou base_score pre XGBoost 2.x."""
    if hasattr(model, "base_score") and isinstance(model.base_score, str):
        model.base_score = 0.5

    booster = model.get_booster()
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(x_data)

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(expected_value[-1])

    return shap_values, expected_value


# -- SHAP VIZUALIZACIE 

def plot_shap_summary(model, x_data, user_id):
    """Globalny SHAP summary plot pre jedneho pouzivatela."""
    print(f"Generujem SHAP analyzu pre pouzivatela {user_id}...")

    try:
        shap_values, _ = compute_shap(model, x_data)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, x_data, show=False)
        plt.title(f"SHAP Top Features - Pouzivatel {user_id}", fontsize=14)
        plt.tight_layout()

        os.makedirs("xai", exist_ok=True)
        plt.savefig(f"xai/shap_summary_user_{user_id}.png", dpi=300)
        plt.show()
        plt.close()
        print(f"Graf ulozeny ako: xai/shap_summary_user_{user_id}.png")

    except Exception as e:
        print(f"SHAP zlyhal: {e}")


def plot_shap_local_explanations(model, x_test, y_test, user_id):
    """Lokalne SHAP vysvetlenia: Force Plot, Decision Plot, Feature Importance."""
    print(f"  [XAI] Generujem lokalne SHAP vysvetlenia pre pouzivatela {user_id}...")

    try:
        shap_values, expected_value = compute_shap(model, x_test)

        pos_idx = np.where(y_test.values == 1)[0]
        neg_idx = np.where(y_test.values == 0)[0]

        os.makedirs("xai", exist_ok=True)

        shap_sums = shap_values.sum(axis=1)

        def _save_force_plot(sample_idx, title, fname, top_n=12):
            sv = shap_values[sample_idx]
            row = x_test.iloc[sample_idx]

            top_mask = np.argsort(np.abs(sv))[-top_n:]
            sv_top = sv[top_mask]
            vals_top = row.iloc[top_mask].round(4)
            names_top = x_test.columns[top_mask].tolist()

            shap.force_plot(
                expected_value,
                sv_top,
                vals_top,
                feature_names=names_top,
                matplotlib=True,
                show=False,
                figsize=(20, 3),
                text_rotation=15,
            )
            plt.suptitle(title, fontsize=11, y=1.05, fontweight="bold")
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()

        # Force plot - pozitivne vzorky
        pos_dir = f"xai/force_plots_user_{user_id}/positive"
        os.makedirs(pos_dir, exist_ok=True)
        print(f"  [XAI] Generujem {len(pos_idx)} force plotov (pozitivne)...")
        for rank, sample_idx in enumerate(pos_idx, start=1):
            shap_sum_val = shap_sums[sample_idx]
            _save_force_plot(
                sample_idx,
                title=(f"Force Plot - U{user_id} | Autentifikovany | "
                       f"vzorka {rank}/{len(pos_idx)}  (SHAP suma: {shap_sum_val:.3f})"),
                fname=f"{pos_dir}/sample_{rank:03d}.png",
            )

        # Force plot - negativne vzorky (max 30)
        neg_dir = f"xai/force_plots_user_{user_id}/negative"
        os.makedirs(neg_dir, exist_ok=True)
        rng = np.random.default_rng(42)
        neg_sample = neg_idx if len(neg_idx) <= 30 else rng.choice(neg_idx, size=30, replace=False)
        neg_sample = np.sort(neg_sample)
        print(f"  [XAI] Generujem {len(neg_sample)} force plotov (negativne)...")
        for rank, sample_idx in enumerate(neg_sample, start=1):
            shap_sum_val = shap_sums[sample_idx]
            _save_force_plot(
                sample_idx,
                title=f"Force Plot - U{user_id} | Impostor | vzorka {rank}/{len(neg_sample)}",
                fname=f"{neg_dir}/sample_{rank:03d}.png",
            )

        # Decision plot
        n_pos = len(pos_idx)
        rng_dp = np.random.default_rng(42)
        sel_neg_dp = np.sort(
            rng_dp.choice(neg_idx, size=min(n_pos, len(neg_idx)), replace=False)
        )
        combined_idx = np.concatenate([pos_idx, sel_neg_dp])
        highlight = list(range(n_pos))

        plt.figure(figsize=(14, 9))
        shap.decision_plot(
            expected_value,
            shap_values[combined_idx],
            x_test.iloc[combined_idx],
            highlight=highlight,
            show=False,
        )
        plt.title(
            f"Decision Plot - Pouzivatel {user_id}\n"
            f"Cervena = autentifikovany, Modra = impostor",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(f"xai/decision_plot_user_{user_id}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Summary plot
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, x_test, show=False, max_display=15)
        plt.title(f"SHAP Summary - Pouzivatel {user_id}", fontsize=13, fontweight="bold", pad=14)
        plt.tight_layout()
        plt.savefig(f"xai/shap_summary_user_{user_id}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Feature importance bar chart (top 5)
        top_n_imp = 5
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        imp_series = (
            pd.Series(mean_abs_shap, index=x_test.columns)
            .sort_values(ascending=True)
            .tail(top_n_imp)
        )

        fig, ax = plt.subplots(figsize=(10, max(6, top_n_imp * 0.38)))
        reversed_colors = plt.cm.magma(1 - (imp_series.values / imp_series.values.max()))

        ax.barh(
            imp_series.index,
            imp_series.values,
            color=reversed_colors,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xlabel("Priemerna |SHAP| hodnota", fontsize=11)
        ax.set_title(f"SHAP Feature Importance - user {user_id}", fontsize=12, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.tick_params(axis="y", labelsize=9)
        plt.tight_layout()
        plt.savefig(f"xai/feature_importance_user_{user_id}.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  [XAI] Grafy ulozene do xai/ pre pouzivatela {user_id}")

    except Exception as e:
        print(f"  [XAI] Zlyhalo pre pouzivatela {user_id}: {e}")


# -- GLOBALNE VIZUALIZACIE --

def analyze_feature_rank_stability(all_importances, feature_names, user_ids,
                                   top_n_print=10, subsets=(5, 10), enabled=True):
    """Spearman rank korelacie poradia crt napriec pouzivatelmi."""
    if not enabled:
        print("[XAI] analyze_feature_rank_stability: VYPNUTE")
        return None

    from scipy.stats import t as t_dist

    os.makedirs("xai", exist_ok=True)

    imp_matrix = np.array(all_importances)
    n_users = imp_matrix.shape[0]
    feature_names = list(feature_names)

    mean_imp = imp_matrix.mean(axis=0)
    global_ranked = np.argsort(mean_imp)[::-1]

    # Vypis poradia per pouzivatel
    print("\n" + "=" * 70)
    print(f"PORADIE CHARAKTERISTIK PER POUZIVATEL (top {top_n_print})")
    print("=" * 70)
    for i, uid in enumerate(user_ids):
        ranked_idx = np.argsort(imp_matrix[i])[::-1][:top_n_print]
        print(f"\n  Pouzivatel {uid}:")
        for rank, j in enumerate(ranked_idx, 1):
            print(f"    {rank:2d}. {feature_names[j]:<40} | gain: {imp_matrix[i][j]:.6f}")

    results = {}

    for k in subsets:
        top_k_idx = global_ranked[:k]
        top_k_names = [feature_names[j] for j in top_k_idx]

        sub_matrix = imp_matrix[:, top_k_idx]

        print(f"\n{'=' * 70}")
        print(f"SPEARMAN RANK KORELACIA - TOP {k} CHARAKTERISTIK")
        print(f"  Crty: {', '.join(top_k_names)}")
        print("=" * 70)

        result = spearmanr(sub_matrix, axis=1)
        corr_matrix = np.array(result.statistic) if hasattr(result, "statistic") \
                      else np.array(result.correlation)

        if corr_matrix.ndim == 0:
            print("  [WARN] Prilis maly subset, preskakujem.")
            continue

        upper_mask = np.triu(np.ones((n_users, n_users), dtype=bool), k=1)
        all_corrs = corr_matrix[upper_mask]
        n_pairs = len(all_corrs)

        df = max(k - 2, 1)
        t_stat = all_corrs * np.sqrt(df / (1 - all_corrs ** 2 + 1e-12))
        pvals = 2 * t_dist.sf(np.abs(t_stat), df=df)
        sig = (pvals < 0.05).sum()

        print(f"  Pocet pouzivatelov:             {n_users}")
        print(f"  Pocet testovanych dvojic:       {n_pairs}")
        print(f"  Priemerna korelacia:            {all_corrs.mean():.4f} +/- {all_corrs.std():.4f}")
        print(f"  Median korelacie:               {np.median(all_corrs):.4f}")
        print(f"  Min / Max:                      {all_corrs.min():.4f} / {all_corrs.max():.4f}")
        print(f"  Statisticky vyznamne (p < 0.05): {sig} / {n_pairs}  ({sig/n_pairs:.1%})")

        # Heatmapa
        annot = n_users <= 25
        fig_sz = max(10, n_users * 0.22)
        fig, ax = plt.subplots(figsize=(fig_sz, fig_sz * 0.85))
        sns.heatmap(
            corr_matrix,
            ax=ax,
            xticklabels=user_ids,
            yticklabels=user_ids,
            cmap="coolwarm",
            vmin=-1, vmax=1,
            annot=annot,
            fmt=".2f" if annot else "",
            square=True,
            linewidths=0.3 if annot else 0,
            cbar_kws={"label": "Spearmanova korelacia"},
        )
        ax.set_title(
            f"Spearman Rank Korelacia - top {k} charakteristik\n"
            f"priemer = {all_corrs.mean():.3f}  |  median = {np.median(all_corrs):.3f}  |"
            f"  {sig}/{n_pairs} parov p < 0.05",
            fontsize=12, fontweight="bold",
        )
        ax.tick_params(axis="both", labelsize=max(5, 9 - n_users // 15))
        plt.tight_layout()
        path_hm = f"xai/feature_rank_stability_top{k}_heatmap.png"
        plt.savefig(path_hm, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [XAI] Heatmapa ulozena: {path_hm}")

        # Histogram
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(all_corrs, bins=40, color="#4472C4", edgecolor="black",
                alpha=0.8, label="Korelacne koeficienty")
        ax.axvline(all_corrs.mean(), color="red", linestyle="--", linewidth=2,
                   label=f"Priemer: {all_corrs.mean():.3f}")
        ax.axvline(np.median(all_corrs), color="orange", linestyle="--", linewidth=2,
                   label=f"Median: {np.median(all_corrs):.3f}")
        ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_xlabel("Spearmanova korelacia poradia crt", fontsize=11)
        ax.set_ylabel("Pocet dvojic pouzivatelov", fontsize=11)
        ax.set_title(
            f"Distribucia Spearmanovej korelacie - top {k} crt  |  {n_pairs} parov\n"
            f"Statisticky vyznamnych (p < 0.05): {sig}/{n_pairs} ({sig/n_pairs:.1%})",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        plt.tight_layout()
        path_hist = f"xai/feature_rank_stability_top{k}_histogram.png"
        plt.savefig(path_hist, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [XAI] Histogram ulozeny: {path_hist}")

        results[k] = {"corr_matrix": corr_matrix, "corrs": all_corrs, "pvals": pvals}

    return results


def plot_violin_user_pairs(df_test, feature_names, mean_importance,
                           n_pairs=3, top_n=8, seed=42, fixed_pairs=None):
    """Split violin plot pre porovnanie rozlozenia top N crt medzi dvojicami pouzivatelov."""
    rng = np.random.default_rng(seed)
    os.makedirs("xai", exist_ok=True)

    imp_series = pd.Series(mean_importance, index=feature_names)
    top_features = imp_series.nlargest(top_n).index.tolist()
    short_labels = {f: (f[:22] + "..." if len(f) > 22 else f) for f in top_features}

    all_user_ids = sorted(df_test["user_id"].unique())

    if fixed_pairs is not None:
        pairs = [tuple(sorted([int(a), int(b)])) for a, b in fixed_pairs]
    else:
        pairs, used, attempts = [], set(), 0
        while len(pairs) < n_pairs and attempts < 1000:
            a, b = rng.choice(all_user_ids, size=2, replace=False)
            key = tuple(sorted([int(a), int(b)]))
            if key not in used:
                pairs.append(key)
                used.add(key)
            attempts += 1

    print(f"\n[XAI] Violin plot - dvojice: {pairs}")

    color_a = "#4472C4"
    color_b = "#ED7D31"

    for uid_a, uid_b in pairs:
        data_a = df_test[df_test["user_id"] == uid_a][top_features]
        data_b = df_test[df_test["user_id"] == uid_b][top_features]

        rows = []
        for feat in top_features:
            combined = pd.concat([data_a[feat].dropna(), data_b[feat].dropna()])
            mu, sigma = combined.mean(), combined.std() + 1e-9
            label = short_labels[feat]
            for val in data_a[feat].dropna().values:
                rows.append({"Priznak": label, "Hodnota": (val - mu) / sigma,
                             "Pouzivatel": f"Pouzivatel {uid_a}"})
            for val in data_b[feat].dropna().values:
                rows.append({"Priznak": label, "Hodnota": (val - mu) / sigma,
                             "Pouzivatel": f"Pouzivatel {uid_b}"})

        df_long = pd.DataFrame(rows)
        feat_order = [short_labels[f] for f in top_features]

        fig, ax = plt.subplots(figsize=(max(10, top_n * 1.6), 7))
        palette = {f"Pouzivatel {uid_a}": color_a, f"Pouzivatel {uid_b}": color_b}

        sns.violinplot(
            data=df_long,
            x="Priznak", y="Hodnota",
            hue="Pouzivatel",
            order=feat_order,
            split=True,
            inner="quartile",
            palette=palette,
            linewidth=1.1,
            ax=ax,
        )

        ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_title(
            f"Pouzivatel {uid_a}  vs  Pouzivatel {uid_b} - rozlozenie top {top_n} crt",
            fontsize=13, fontweight="bold", pad=14,
        )
        ax.set_xlabel("Biometricky priznak", fontsize=11)
        ax.set_ylabel("Normalizovana hodnota (z-score)", fontsize=11)
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(title="", fontsize=10, framealpha=0.9)

        plt.tight_layout()
        fname = f"xai/violin_user_{uid_a}_vs_{uid_b}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [XAI] Ulozeny: {fname}")


def plot_global_feature_importance(all_importances, feature_names):
    """Priemerna dolezitost vsetkych priznakov napriec modelmi."""
    df_imp = pd.DataFrame(all_importances, columns=feature_names)
    mean_imp = df_imp.mean().sort_values(ascending=False)

    plt_height = max(6, len(feature_names) * 0.3)
    plt.figure(figsize=(12, plt_height))

    sns.barplot(x=mean_imp.values, y=mean_imp.index, palette="magma")

    plt.title(f"Globalna dolezitost crt (Priemer cez {len(all_importances)} pouzivatelov)", fontsize=16)
    plt.xlabel("Priemerny prinos (Gain Importance)", fontsize=12)
    plt.ylabel("Nazov biometrickeho priznaku", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()

    os.makedirs("xai", exist_ok=True)
    plt.savefig("xai/global_feature_importance_ALL.png", dpi=300)
    plt.show()

    print("\n" + "=" * 50)
    print("KOMPLETNY REBRICEK DOLEZITOSTI:")
    print("=" * 50)
    for i, (name, val) in enumerate(mean_imp.items(), 1):
        print(f"{i:3d}. {name:30s} | Gain: {val:.6f}")
    print("=" * 50)


def plot_eer_histogram(df_results, max_acceptable_eer):
    plt.figure(figsize=(10, 6))
    sns.histplot(df_results["EER"], bins=500, kde=True, color="skyblue", edgecolor="black")

    avg_eer = df_results["EER"].mean()
    plt.axvline(avg_eer, color="red", linestyle="dashed", linewidth=2,
                label=f"Priemerne EER: {avg_eer:.4f}")
    plt.axvline(max_acceptable_eer, color="orange", linestyle="dashed", linewidth=2,
                label=f"Max akceptovatelne: {max_acceptable_eer}")

    plt.title("Rozlozenie Equal Error Rate (EER) napriec pouzivatelmi", fontsize=14)
    plt.xlabel("EER (Hodnota)", fontsize=12)
    plt.ylabel("Pocet pouzivatelov", fontsize=12)
    plt.legend()
    plt.grid(axis="y", alpha=0.75)

    plt.savefig("xai/eer_histogram.png", dpi=300)
    plt.show()
    plt.close()


def plot_sorted_eer(df_results):
    sorted_eer = df_results["EER"].sort_values().values
    user_index = range(1, len(sorted_eer) + 1)

    plt.figure(figsize=(10, 6))
    plt.scatter(user_index, sorted_eer, color="teal", s=15, label="EER pouzivatela")
    plt.fill_between(user_index, sorted_eer, color="teal", alpha=0.1)

    plt.axhline(y=0.15, color="orange", linestyle="--", label="Prah 0.15")

    plt.title("EER zoradene podla kvality modelu (1 = najlepsi)")
    plt.xlabel("Poradie pouzivatela")
    plt.ylabel("EER")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_eer_boxplot(df_results, max_acceptable_eer):
    plt.figure(figsize=(8, 6))

    plt.boxplot(df_results["EER"], vert=True, patch_artist=True,
                widths=0.5,
                boxprops=dict(facecolor="lightblue", color="darkblue"),
                medianprops=dict(color="red", linewidth=2),
                whiskerprops=dict(color="darkblue"),
                capprops=dict(color="darkblue"),
                flierprops=dict(marker="o", markerfacecolor="orange",
                               markersize=8, linestyle="none"))

    plt.axhline(y=max_acceptable_eer, color="orange", linestyle="--",
                linewidth=2, label=f"Prah: {max_acceptable_eer}")

    avg_eer = df_results["EER"].mean()
    plt.axhline(y=avg_eer, color="green", linestyle=":",
                linewidth=2, label=f"Priemer: {avg_eer:.4f}")

    plt.ylabel("EER", fontsize=12)
    plt.title("Boxplot EER hodnot napriec pouzivatelmi", fontsize=14, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig("xai/eer_boxplot.png", dpi=300)
    plt.show()
    plt.close()


# -- TRENINGOVA SLUCKA --

def train_all_users(df_train, df_test, columns_to_drop, users_for_xai):
    """Natrenuje binarny model pre kazdeho pouzivatela a vrati vysledky."""
    all_users = sorted(df_train["user_id"].unique())
    global_results = []
    all_importances = []

    for target_user_id in all_users:
        df_train_user = df_train.copy()
        df_test_user = df_test.copy()

        df_train_user["label"] = (df_train_user.user_id == target_user_id).astype(int)
        df_test_user["label"] = (df_test_user.user_id == target_user_id).astype(int)

        df_train_full = df_train_user.sample(frac=1, random_state=42).reset_index(drop=True)

        x_train = df_train_full.drop(columns=columns_to_drop + ["label"], errors="ignore")
        y_train = df_train_full["label"]

        x_test = df_test_user.drop(columns=columns_to_drop + ["label"], errors="ignore")
        y_test = df_test_user["label"]

        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count

        if y_test.sum() == 0 or pos_count == 0:
            continue

        scale_pos_weight = neg_count / pos_count
        clf = build_model(scale_pos_weight)
        clf.fit(x_train, y_train)

        all_importances.append(clf.feature_importances_)

        y_proba = clf.predict_proba(x_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        fnr = 1 - tpr
        eer_index = np.nanargmin(np.abs(fnr - fpr))
        eer = fpr[eer_index]
        eer_threshold = thresholds[eer_index]

        y_pred = (y_proba >= eer_threshold).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        global_results.append({
            "user_id": target_user_id,
            "AUC": roc_auc,
            "EER": eer,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
        })

        print(f"Pouzivatel {target_user_id:3d} | "
              f"pos:{int(pos_count)} neg:{int(neg_count)} "
              f"| AUC: {roc_auc:.4f} | EER: {eer:.6f}")

        if target_user_id in users_for_xai:
            plot_shap_local_explanations(clf, x_test, y_test, target_user_id)

    return pd.DataFrame(global_results), all_importances, x_train.columns


def print_evaluation_summary(df_results, max_acceptable_eer):
    """Vypise sumarnu tabulku vysledkov."""
    total = len(df_results)
    unacceptable = (df_results["EER"] > max_acceptable_eer).sum()

    print("\n evaluation ")
    print("." * 70)
    print(f"num users: {total}")
    print(f"{'-' * 70}")
    print(f"{'-':<15} | {'avg':>8} | {'std':>8} | {'med':>8} | {'min':>8} | {'max':>8}")
    print(f"{'-' * 70}")

    for col in ["AUC", "EER", "Accuracy", "Precision", "Recall"]:
        s = df_results[col]
        print(f"{col:<15} | {s.mean():8.4f} | {s.std():8.4f} | {s.median():8.4f} "
              f"| {s.min():8.4f} | {s.max():8.4f}")

    print(f"{'-' * 70}")
    print(f"Pouzivatelia s nepripustnym EER (> {max_acceptable_eer:.2f}): "
          f"{unacceptable} ({unacceptable/total:.1%})")
    print("=" * 70)




if __name__ == "__main__":
    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)

    df_train, df_test = remove_users(df_train, df_test, USERS_TO_DROP)

    df_results, all_importances, feature_cols = train_all_users(
        df_train, df_test, COLUMNS_TO_DROP, USERS_FOR_XAI
    )

    print_evaluation_summary(df_results, MAX_ACCEPTABLE_EER)

    plot_global_feature_importance(all_importances, feature_cols)
    plot_eer_boxplot(df_results, MAX_ACCEPTABLE_EER)

    analyze_feature_rank_stability(
        all_importances,
        feature_names=feature_cols,
        user_ids=df_results["user_id"].tolist(),
        top_n_print=10,
        subsets=(5, 10),
        enabled=True,
    )

    mean_imp = np.array(all_importances).mean(axis=0)
    plot_violin_user_pairs(
        df_test,
        feature_names=list(feature_cols),
        mean_importance=mean_imp,
        top_n=8,
        fixed_pairs=[(80, 86)],
    )
