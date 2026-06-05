import pandas as pd
import numpy as np

def aggregate_global_user_features(df):
    """
    Vypočíta globálne agregované črty (mean, std) pre každého používateľa 
    na základe jeho všetkých historických segmentov.
    """
    # Pracujeme s kópiou, aby sme neupravovali pôvodný DataFrame
    df_agg = df.copy()

    # 1. Ošetrenie click_duration (nahradenie 0 mediánom, individuálne pre každého používateľa)
    if 'click_duration' in df_agg.columns:
        def replace_zero_with_median(series):
            non_zero_clicks = series[series > 0]
            if not non_zero_clicks.empty:
                return series.replace(0, non_zero_clicks.median())
            return series # Ak nemá žiadne kliky, vráti pôvodnú sériu
            
        # Transform aplikuje funkciu na každú user_id skupinu zvlášť
        df_agg['click_duration'] = df_agg.groupby('user_id')['click_duration'].transform(replace_zero_with_median)

    # 2. Určenie stĺpcov, ktoré ideme agregovať
    # Vylúčime identifikátory, kategórie a textové stĺpce
    exclude_cols = ['user_id', 'segment_id', 'is_pc', 'is_dd', 'csv_file', 'window_id']
    
    # Vyberieme len numerické stĺpce, ktoré sa nenachádzajú v exclude_cols
    feature_cols = [c for c in df_agg.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_agg[c])]

    # 3. Zoskupenie (groupby) podľa používateľa a výpočet mean a std naraz
    aggregated = df_agg.groupby('user_id')[feature_cols].agg(['mean', 'std'])

    # 4. Upratanie názvov stĺpcov (odstránenie MultiIndex štruktúry)
    aggregated.columns = [f"{col}_{func}" for col, func in aggregated.columns]

    # 5. Vytiahnutie user_id z indexu späť do normálneho stĺpca
    aggregated = aggregated.reset_index()
    
    # Voliteľné: Uloženie do CSV
    # aggregated.to_csv("global_aggregated_features.csv", index=False)

    return aggregated

features = pd.read_csv("test_data.csv")
features_agg = aggregate_global_user_features(features)
features_agg.to_csv("test_data_ag.csv", index=False)


