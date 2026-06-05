import os
import pandas as pd
import numpy as np

import shap
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
from scipy.stats import spearmanr

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

def analyze_feature_rank_stability(all_importances, feature_names, user_ids,
                                    top_n_print=10, subsets=(5, 10), enabled=True):
    """
    Globálna stabilita poradia čŕt naprieč používateľmi.

    1. Pre každého používateľa vypíše top-N najdôležitejších charakteristík.
    2. Spearman rank correlation sa počíta IBA pre top-K čŕty (podľa mean importance),
       kde K je každá hodnota zo zoznamu `subsets` (napr. 5 a 10).
    3. Pre každý subset: heatmapa + histogram distribúcie korelácií.

    Parametre:
        all_importances – list gain-importance vektorov (jeden per používateľ)
        feature_names   – názvy príznakov
        user_ids        – zoznam user_id v rovnakom poradí ako all_importances
        top_n_print     – koľko top čŕt vypísať per používateľ do konzoly
        subsets         – tuple veľkostí podmnožín pre Spearman test, napr. (5, 10)
        enabled         – prepínač: False = preskočí bez výpočtu
    """
    if not enabled:
        print("[XAI] analyze_feature_rank_stability: VYPNUTÉ (enabled=False)")
        return None

    from scipy.stats import t as t_dist

    os.makedirs("xai", exist_ok=True)

    imp_matrix    = np.array(all_importances)   # (n_users, n_features)
    n_users       = imp_matrix.shape[0]
    feature_names = list(feature_names)

    # Globálne poradie čŕt podľa priemeru cez všetkých používateľov
    mean_imp      = imp_matrix.mean(axis=0)
    global_ranked = np.argsort(mean_imp)[::-1]   # indexy od najdôležitejšej

    # ------------------------------------------------------------------ #
    # 1. Výpis poradia per používateľ                                      #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print(f"PORADIE CHARAKTERISTÍK PER POUŽÍVATEĽ (top {top_n_print})")
    print("=" * 70)
    for i, uid in enumerate(user_ids):
        ranked_idx = np.argsort(imp_matrix[i])[::-1][:top_n_print]
        print(f"\n  Používateľ {uid}:")
        for rank, j in enumerate(ranked_idx, 1):
            print(f"    {rank:2d}. {feature_names[j]:<40} | gain: {imp_matrix[i][j]:.6f}")

    # ------------------------------------------------------------------ #
    # 2. Spearman korelácia — pre každý subset zvlášť                     #
    # ------------------------------------------------------------------ #
    results = {}

    for k in subsets:
        # Indexy top-K čŕt podľa globálneho priemeru
        top_k_idx  = global_ranked[:k]
        top_k_names = [feature_names[j] for j in top_k_idx]

        # Podmatrica: (n_users × k) — len vybrané čŕty
        sub_matrix = imp_matrix[:, top_k_idx]

        print(f"\n{'=' * 70}")
        print(f"SPEARMAN RANK KORELÁCIA — TOP {k} CHARAKTERISTÍK")
        print(f"  Čŕty: {', '.join(top_k_names)}")
        print("=" * 70)

        # Korelácia naraz pre celú maticu (scipy)
        result      = spearmanr(sub_matrix, axis=1)
        corr_matrix = np.array(result.statistic) if hasattr(result, "statistic") \
                      else np.array(result.correlation)

        # Ak k=1, spearmanr vráti skalár — ochrana
        if corr_matrix.ndim == 0:
            print("  [WARN] Príliš malý subset, preskakujem.")
            continue

        upper_mask = np.triu(np.ones((n_users, n_users), dtype=bool), k=1)
        all_corrs  = corr_matrix[upper_mask]
        n_pairs    = len(all_corrs)

        # P-hodnoty (df = k - 2)
        df     = max(k - 2, 1)
        t_stat = all_corrs * np.sqrt(df / (1 - all_corrs ** 2 + 1e-12))
        pvals  = 2 * t_dist.sf(np.abs(t_stat), df=df)
        sig    = (pvals < 0.05).sum()

        print(f"  Počet používateľov:             {n_users}")
        print(f"  Počet testovaných dvojíc:       {n_pairs}")
        print(f"  Priemerná korelácia:            {all_corrs.mean():.4f} ± {all_corrs.std():.4f}")
        print(f"  Medián korelácie:               {np.median(all_corrs):.4f}")
        print(f"  Min / Max:                      {all_corrs.min():.4f} / {all_corrs.max():.4f}")
        print(f"  Štatist. významné (p < 0.05):  {sig} / {n_pairs}  ({sig/n_pairs:.1%})")

        # --- Heatmapa ---
        annot  = n_users <= 25
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
            cbar_kws={"label": "Spearmanova korelácia"},
        )
        ax.set_title(
            f"Spearman Rank Korelácia — top {k} charakteristík\n"
            f"priemer = {all_corrs.mean():.3f}  |  medián = {np.median(all_corrs):.3f}  |"
            f"  {sig}/{n_pairs} párov p < 0.05",
            fontsize=12, fontweight="bold",
        )
        ax.tick_params(axis="both", labelsize=max(5, 9 - n_users // 15))
        plt.tight_layout()
        path_hm = f"xai/feature_rank_stability_top{k}_heatmap.png"
        plt.savefig(path_hm, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [XAI] Heatmapa uložená: {path_hm}")

        # --- Histogram ---
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(all_corrs, bins=40, color="#4472C4", edgecolor="black",
                alpha=0.8, label="Korelačné koeficienty")
        ax.axvline(all_corrs.mean(),     color="red",    linestyle="--", linewidth=2,
                   label=f"Priemer: {all_corrs.mean():.3f}")
        ax.axvline(np.median(all_corrs), color="orange", linestyle="--", linewidth=2,
                   label=f"Medián: {np.median(all_corrs):.3f}")
        ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_xlabel("Spearmanova korelácia poradia čŕt", fontsize=11)
        ax.set_ylabel("Počet dvojíc používateľov",         fontsize=11)
        ax.set_title(
            f"Distribúcia Spearmanovej korelácie — top {k} čŕt  |  {n_pairs} párov\n"
            f"Štatisticky významných (p < 0.05): {sig}/{n_pairs} ({sig/n_pairs:.1%})",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        plt.tight_layout()
        path_hist = f"xai/feature_rank_stability_top{k}_histogram.png"
        plt.savefig(path_hist, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [XAI] Histogram uložený: {path_hist}")

        results[k] = {"corr_matrix": corr_matrix, "corrs": all_corrs, "pvals": pvals}

    return results


def plot_violin_user_pairs(df_test, feature_names, mean_importance,
                           n_pairs=3, top_n=8, seed=42, fixed_pairs=None):
    """
    Split violin plot: porovnanie rozloženia top N čŕt pre dvojice používateľov.
    Každý pár = 1 graf, X-os = čŕty, ľavá polovica = user A, pravá = user B.
    Hodnoty sú z-score normalizované pre porovnateľnosť naprieč čŕtami.

    fixed_pairs – ak je zadaný (napr. [(80, 86)]), použije tieto páry namiesto náhodných.
    """
    rng = np.random.default_rng(seed)
    os.makedirs("xai", exist_ok=True)

    # --- Top N čŕt podľa priemernej gain dôležitosti ---
    imp_series = pd.Series(mean_importance, index=feature_names)
    top_features = imp_series.nlargest(top_n).index.tolist()

    # Skrátenie dlhých názvov čŕt pre os X
    short_labels = {f: (f[:22] + "…" if len(f) > 22 else f) for f in top_features}

    all_user_ids = sorted(df_test["user_id"].unique())

    # --- Výber dvojíc: pevné alebo náhodné ---
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

    print(f"\n[XAI] Violin plot — dvojice: {pairs}")

    color_a = "#4472C4"   # modrá
    color_b = "#ED7D31"   # oranžová (ako v príklade)

    for uid_a, uid_b in pairs:
        data_a = df_test[df_test["user_id"] == uid_a][top_features]
        data_b = df_test[df_test["user_id"] == uid_b][top_features]

        # --- Dlhý formát s z-score normalizáciou per-feature ---
        rows = []
        for feat in top_features:
            combined = pd.concat([data_a[feat].dropna(), data_b[feat].dropna()])
            mu, sigma = combined.mean(), combined.std() + 1e-9
            label = short_labels[feat]
            for val in data_a[feat].dropna().values:
                rows.append({"Príznak": label, "Hodnota": (val - mu) / sigma,
                             "Používateľ": f"Používateľ {uid_a}"})
            for val in data_b[feat].dropna().values:
                rows.append({"Príznak": label, "Hodnota": (val - mu) / sigma,
                             "Používateľ": f"Používateľ {uid_b}"})

        df_long = pd.DataFrame(rows)
        # Zachovanie poradia čŕt podľa dôležitosti
        feat_order = [short_labels[f] for f in top_features]

        fig, ax = plt.subplots(figsize=(max(10, top_n * 1.6), 7))

        palette = {f"Používateľ {uid_a}": color_a, f"Používateľ {uid_b}": color_b}

        try:
            # seaborn ≥ 0.12 — split parameter
            sns.violinplot(
                data=df_long,
                x="Príznak", y="Hodnota",
                hue="Používateľ",
                order=feat_order,
                split=True,
                inner="quartile",
                palette=palette,
                linewidth=1.1,
                ax=ax,
            )
        except TypeError:
            # fallback pre staršie verzie seaborn
            sns.violinplot(
                data=df_long,
                x="Príznak", y="Hodnota",
                hue="Používateľ",
                order=feat_order,
                inner="quartile",
                palette=palette,
                linewidth=1.1,
                ax=ax,
            )

        ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_title(
            f"Používateľ {uid_a}  vs  Používateľ {uid_b} — rozloženie top {top_n} čŕt",
            fontsize=13, fontweight="bold", pad=14,
        )
        ax.set_xlabel("Biometrický príznak", fontsize=11)
        ax.set_ylabel("Normalizovaná hodnota (z-score)", fontsize=11)
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(title="", fontsize=10, framealpha=0.9)

        plt.tight_layout()
        fname = f"xai/violin_user_{uid_a}_vs_{uid_b}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [XAI] Uložený: {fname}")


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

def plot_shap_local_explanations(model, x_test, y_test, user_id):
    """
    H2 – Lokálne SHAP vysvetlenia: Force Plot + Decision Plot.
    Pre každého používateľa vygeneruje:
      - Force plot pre autentifikovaný pohyb (label=1)
      - Force plot pre neautentifikovaný pohyb (impostor, label=0)
      - Decision plot pre prvých 5 vzoriek z každej triedy
    """
    print(f"  [XAI] Generujem lokálne SHAP vysvetlenia pre používateľa {user_id}...")

    try:
        if hasattr(model, "base_score") and isinstance(model.base_score, str):
            model.base_score = 0.5

        booster = model.get_booster()
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(x_test)

        # expected_value môže byť list pri niektorých verziách SHAP – berieme skalár
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = float(expected_value[-1])

        pos_idx = np.where(y_test.values == 1)[0]
        neg_idx = np.where(y_test.values == 0)[0]

        os.makedirs("xai", exist_ok=True)

        # Celkový SHAP príspevok (suma) pre každú vzorku
        shap_sums = shap_values.sum(axis=1)

        # Farby: modrá = legitímny používateľ, červená = impostor
        #CLR_LEGIT   = "#0089fa"
        #CLR_IMPOSTOR = "#fb0062"

        def _save_force_plot(sample_idx, title, fname, top_n=12):
            """
            Force plot pre jednu vzorku.
            Modrá = features tlačiace k autentifikácii (legitímny smer).
            Červená = features tlačiace k odmietnutiu (impostor smer).
            """
            sv  = shap_values[sample_idx]
            row = x_test.iloc[sample_idx]

            top_mask  = np.argsort(np.abs(sv))[-top_n:]
            sv_top    = sv[top_mask]
            vals_top  = row.iloc[top_mask].round(4)
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
                # [0] = "high" smer (pozitívny SHAP → autentifikácia) = modrá
                # [1] = "low"  smer (negatívny SHAP → odmietnutie)    = červená
                #plot_cmap=[ CLR_IMPOSTOR, CLR_LEGIT]
            )
            plt.suptitle(title, fontsize=11, y=1.05, fontweight="bold")
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()

        # --- Force Plot: VŠETKY autentifikované vzorky ---
        pos_dir = f"xai/force_plots_user_{user_id}/positive"
        os.makedirs(pos_dir, exist_ok=True)
        print(f"  [XAI] Generujem {len(pos_idx)} force plotov (pozitívne)...")
        for rank, sample_idx in enumerate(pos_idx, start=1):
            shap_sum_val = shap_sums[sample_idx]
            _save_force_plot(
                sample_idx,
                title=(f"Force Plot – U{user_id} | Autentifikovaný | "
                       f"vzorka {rank}/{len(pos_idx)}  (SHAP suma: {shap_sum_val:.3f})"),
                fname=f"{pos_dir}/sample_{rank:03d}.png",
            )

        # --- Force Plot: VŠETKY impostor vzorky ---
        # Impostori môžu byť stovky – limitujeme na 30 náhodných pre prehľadnosť
        neg_dir = f"xai/force_plots_user_{user_id}/negative"
        os.makedirs(neg_dir, exist_ok=True)
        rng = np.random.default_rng(42)
        neg_sample = neg_idx if len(neg_idx) <= 30 else rng.choice(neg_idx, size=30, replace=False)
        neg_sample = np.sort(neg_sample)
        print(f"  [XAI] Generujem {len(neg_sample)} force plotov (negatívne)...")
        for rank, sample_idx in enumerate(neg_sample, start=1):
            shap_sum_val = shap_sums[sample_idx]
            _save_force_plot(
                sample_idx,
                title=(f"ForcPoužívateľ {user_id} "),
                fname=f"{neg_dir}/sample_{rank:03d}.png",
            )

        # --- Decision Plot: VŠETKY legitímne + rovnaký počet impostorov ---
        # Poradie v poli: najprv všetky pozitívne, potom negatívne
        # → vieme presne, ktoré línie sú ktoré pri následnom presfarbení
        n_pos = len(pos_idx)
        rng_dp = np.random.default_rng(42)
        sel_neg_dp = np.sort(
            rng_dp.choice(neg_idx, size=min(n_pos, len(neg_idx)), replace=False)
        )
        combined_idx = np.concatenate([pos_idx, sel_neg_dp])

        # highlight = pozície v combined_idx ktoré sú pozitívne (legitímny používateľ)
        # combined_idx = [pos_idx..., sel_neg_dp...] → prvých n_pos je vždy pozitívnych
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
            f"Decision Plot – Používateľ {user_id}\n"
            f"Červená = autentifikovaný, Modrá = impostor",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(
            f"xai/decision_plot_user_{user_id}.png",
            dpi=150, bbox_inches="tight"
        )
        plt.close()

        # --- SHAP Summary Plot ---
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            x_test,
            show=False,
            max_display=15,
        )
        plt.title(
            f"SHAP Summary – Používateľ {user_id}",
            fontsize=13, fontweight="bold", pad=14,
        )
        plt.tight_layout()
        plt.savefig(
            f"xai/shap_summary_user_{user_id}.png",
            dpi=150, bbox_inches="tight"
        )
        plt.close()

        # --- Per-user SHAP Feature Importance (bar chart, top 20) ---
        # Priemerná |SHAP| hodnota = koľko každá čŕt v priemere prispela k rozhodnutiu
        # pre TOHTO konkrétneho používateľa (nie globálny priemer)
        top_n_imp = 5
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        imp_series = (
            pd.Series(mean_abs_shap, index=x_test.columns)
            .sort_values(ascending=True)
            .tail(top_n_imp)
        )

        fig, ax = plt.subplots(figsize=(10, max(6, top_n_imp * 0.38)))
        
        # Výpočet obrátenej farby: 1 - (hodnota / max)
        # Tým dosiahneme, že najvyššie hodnoty (hore) dostanú farbu blízku 0 (tmavá)
        # a najnižšie hodnoty (dole) dostanú farbu blízku 1 (svetlá)
        reversed_colors = plt.cm.magma(1 - (imp_series.values / imp_series.values.max()))

        bars = ax.barh(
            imp_series.index,
            imp_series.values,  
            color=reversed_colors,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xlabel("Priemerná |SHAP| hodnota", fontsize=11)
        ax.set_title(
            f"SHAP Feature Importance – user {user_id}\n",
            fontsize=12, fontweight="bold",
        )
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.tick_params(axis="y", labelsize=9)
        plt.tight_layout()
        plt.savefig(
            f"xai/feature_importance_user_{user_id}.png",
            dpi=150, bbox_inches="tight"
        )
        plt.close()

        print(f"  [XAI] Grafy uložené do xai/ pre používateľa {user_id}")

    except Exception as e:
        print(f"  [XAI] Zlyhalo pre používateľa {user_id}: {e}")


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

# data loading
df_train = pd.read_csv("train-ws22-st4.csv")
df_test = pd.read_csv("test-ws22-st4.csv")


users_to_drop = [4, 103]
#users_to_drop = []
df_train, df_test = remove_users(df_train, df_test, users_to_drop)

columns_to_drop = ["user_id", "csv_file", "window_id", "std_angle_std", "std_angle_mean", "scattering_coefficient_norm_std", "scattering_coefficient_norm_mean", "tcm_norm_std", "std_dt_std"]
#columns_to_drop = ["user_id", "csv_file", "window_id"]

all_users = sorted(df_train["user_id"].unique())

# Mediánoví používatelia podľa EER – pre lokálne XAI vysvetlenia
#users_for_xai = {15,80, 90}
users_for_xai = {}
MAX_ACCEPTABLE_EER = 0.15
global_results = []

all_importances = []
for target_user_id in all_users:

    #if target_user_id != 90 and target_user_id != 15 and target_user_id != 80:
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
    
    if target_user_id in users_for_xai:
        plot_shap_local_explanations(xgb, x_test, y_test, target_user_id)



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

# --- Stabilita poradia charakteristík (Spearman rank correlation) ---
# Prepínač: True = spusti analýzu, False = preskočí bez výpočtu
FEATURE_STABILITY_ENABLED = True

analyze_feature_rank_stability(
    all_importances,
    feature_names=x_train.columns,
    user_ids=df_results["user_id"].tolist(),
    top_n_print=10,
    subsets=(5, 10),          # Spearman sa počíta len pre top-5 a top-10 čŕty
    enabled=FEATURE_STABILITY_ENABLED,
)

# --- Violin plot: mediánoví používatelia (user80 vs user86) ---
mean_imp = np.array(all_importances).mean(axis=0)
plot_violin_user_pairs(
    df_test,
    feature_names=list(x_train.columns),
    mean_importance=mean_imp,
    top_n=8,
    fixed_pairs=[(80, 86)],
)  