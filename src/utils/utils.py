import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#  ---------------- DEBUG ----------------
def write_all_segments_to_csv(segments, filename="trajectory_segments.csv"):

    combined = []
    for i, seg in enumerate(segments):
        seg_copy = seg.copy()
        seg_copy['segment_id'] = i  # segment index
        combined.append(seg_copy)

    if combined:
        out_df = pd.concat(combined, ignore_index=True)
        # put segment_id as first column
        cols = ['segment_id'] + [c for c in out_df.columns if c != 'segment_id']
        out_df = out_df[cols]
        out_df.to_csv(filename, index=False)
    else:
        # write empty file with no rows
        pd.DataFrame().to_csv(filename, index=False)





def plot_trajectories(df_trajectory):
    color = 'blue' if df_trajectory['state'].iloc[0] == 'Move' else 'red'
    # X osa = lokálny index trajektórie
    x = range(len(df_trajectory))

    plt.plot(x, df_trajectory['vel'], color=color, alpha=0.2)

    
    plt.ylabel("Rýchlosť")
    plt.title("Rýchlosti trajektórií myši")
    plt.grid(True)
    plt.legend(["Move", "Drag"])

    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_histogram(df, column, max_quantile=0.999):
    """
    Vykreslí histogram pre ľubovoľný stĺpec z DataFrame.
    
    Parametre:
    - df: Vstupný DataFrame
    - column: Názov stĺpca (string)
    - max_quantile: Odreže extrémne outlierov (napr. 0.99 odreže horné 1% hodnôt), 
                    aby bol graf čitateľný. Ak chceš vidieť všetko, nastav na 1.0.
    """
    if column not in df.columns:
        print(f"Chyba: Stĺpec '{column}' sa v DataFrame nenachádza.")
        return

    # Odstránime NaN hodnoty, ktoré by rozbili histogram
    data = df[column].dropna()
    
    # Odrezanie extrémnych hodnôt (outlierov) pre lepšiu vizualizáciu
    limit = data.quantile(max_quantile)
    filtered_data = data[data <= limit]

    plt.figure(figsize=(8, 4))
    
    # Vykreslenie histogramu
    sns.histplot(filtered_data, bins=80, kde=True, color='royalblue', edgecolor='black', alpha=0.7)
    
    if column == 'time_duration':
        plt.axvline(x=200, color='red', linestyle='--', linewidth=2, label='Limit: 200')

    plt.title(f'Distribúcia hodnôt v stĺpci: {column} (do {max_quantile*100:.0f}. percentilu)')
    plt.xlabel(f'Hodnota ({column})')
    plt.ylabel('Počet výskytov')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.show()

    # Základné štatistické info pre kontext
    print(f"--- Štatistika pre stĺpec: {column} ---")
    print(data.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

# Príklady použitia:
# plot_feature_histogram(df, 'dt')
# plot_feature_histogram(df, 'vel')
# plot_feature_histogram(df, 'angle_change')


def remove_users(df_train, df_test, users_to_remove):
    """
    Vymaže zoznam používateľov z trénovacích aj testovacích dát.
    """
    before_train = len(df_train)
    before_test = len(df_test)
    
    # Filtrovanie pomocou operátora ~ (not) a isin()
    df_train_cleaned = df_train[~df_train["user_id"].isin(users_to_remove)].copy()
    df_test_cleaned = df_test[~df_test["user_id"].isin(users_to_remove)].copy()
    
    print(f"--- ČISTENIE DÁT ---")
    print(f"Odstraňujem používateľov: {users_to_remove}")
    print(f"Train: odstránených {before_train - len(df_train_cleaned)} riadkov.")
    print(f"Test:  odstránených {before_test - len(df_test_cleaned)} riadkov.")
    print(f"--------------------\n")
    
    return df_train_cleaned, df_test_cleaned