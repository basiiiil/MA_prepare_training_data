"""
Dieses Script soll alle Prozessschritte der Datenbereinigung und -zusammenführung nacheinander
ausführen, sodass am Ende nur dieses Script bedient werden muss.
"""
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from functions_diagnosen import get_prozedur_charlson_pivot
from functions_labor import get_prozedur_labor_pivot
from util_functions import concat_csv_files


def get_now_label():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M_")

def get_labeled_prozeduren_from_file():
    df_befunde_labeled = concat_csv_files(
        folder_path='befunde_mit_label',
        csv_dtype=None,
    )

    df_befunde_labeled['prozedur_datetime'] = pd.to_datetime(df_befunde_labeled['prozedur_datetime'])
    # df_befunde_labeled['geschlecht'] = df_befunde_labeled['geschlecht'].astype('category')
    df_befunde_labeled['alter_bei_prozedur'] = df_befunde_labeled['alter_bei_prozedur'].astype(int)
    # df_befunde_labeled['predicted_label'] = df_befunde_labeled['predicted_label'].astype('category')

    return df_befunde_labeled

def create_laborwerte_violinplot(ddf_labor):
    # 1. Berechne Anzahl und Mittelwert für jeden Parameter
    df_analyse = ddf_labor[
        ['parameterid_effektiv', 'stunden_vor_prozedur']
    ].copy().compute()
    zeitverteilung_stats = df_analyse.groupby(
        by='parameterid_effektiv',
    )[['stunden_vor_prozedur']].agg(['mean', 'count'])

    # 5. Ergebnis berechnen und anzeigen
    zeitverteilung_stats.columns = ['mittlere_stunden_vor_prozedur', 'anzahl']  # Spaltennamen vereinfachen
    zeitverteilung_stats = zeitverteilung_stats.sort_values(by='anzahl', ascending=False)

    parameter = zeitverteilung_stats.index.tolist()
    # Sortiere die Kategorien im DataFrame nach der Häufigkeit für einen sauberen Plot
    df_analyse['parameterid_effektiv'] = pd.Categorical(
        df_analyse['parameterid_effektiv'], categories=parameter, ordered=True
    )
    df_plot = df_analyse.sort_values('parameterid_effektiv')

    # Erstelle die Visualisierung
    plt.figure(figsize=(12, 32))  # Passe die Größe bei Bedarf an
    sns.violinplot(
        data=df_plot,
        y='parameterid_effektiv',
        x='stunden_vor_prozedur',
        orient='h'  # Horizontale Ausrichtung
    )

    plt.title('Zeitliche Verteilung der Laborparameter vor der Prozedur (Top 20)', fontsize=16)
    plt.xlabel('Stunden vor der Prozedur', fontsize=12)
    plt.ylabel('Laborparameter', fontsize=12)
    plt.gca().invert_xaxis()  # Zeitachse umkehren (0 ist rechts)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()  # Stellt sicher, dass alles gut lesbar ist

    # Speichere die Grafik als Datei
    output_filename = "laborparameter_verteilung_violinplot.png"
    plt.savefig(output_filename)
    plt.close()
    print(f"\nGrafik wurde erfolgreich als '{output_filename}' gespeichert.")

def get_final_pivot_table(ddf):
    # Wir wollen eine Tabelle, in der jede Zeile eine Prozedur ist (identifiziert durch 'Fallnummer' und
    # 'prozedur_datetime') und jede Spalte ein Laborparameter.
    # Der Wert in der Zelle ist der letzte bekannte 'ergebniswert_num'.
    # Zusätzliche Spalten sind 'alter_bei_prozedur', 'geschlecht' und 'lae_kategorie'.

    # 1. Entferne unnötige Spalten, um den Pivot zu beschleunigen
    ddf_pivot_input = ddf[
        [
            'Fallnummer',
            'prozedur_datetime',
            'alter_bei_prozedur',
            'geschlecht',
            'parameterid_effektiv',
            'ergebniswert_num',
            # 'minuten_vor_prozedur',
        ]
    ]

    # 2. Führe den Pivot durch
    #   - index='Fallnummer': Jede Zeile ist ein einzigartiger Fall.
    #   - columns='parameterid_effektiv': Jede Spalte ist ein einzigartiger Laborparameter.
    #   - values='ergebniswert_num': Die Zellen enthalten den numerischen Ergebniswert.
    df_final_features = ddf_pivot_input.pivot_table(
        index='Fallnummer',
        columns='parameterid_effektiv',
        values='ergebniswert_num'
    ).compute()

    print("\nFeature-Tabelle mit den letzten Laborwerten (Ausschnitt):")
    print(df_final_features.head())

    return df_final_features

    # Diese Tabelle 'df_final_features' ist jetzt bereit für das Training deines ML-Modells.
    # Jede Zeile ist ein Sample (ein Fall) und jede Spalte ist ein Feature (ein Laborparameter).
    # Die NaN-Werte bedeuten, dass für diesen Fall dieser Laborparameter nicht gemessen wurde.
    # Diese Lücken musst du im nächsten Schritt mit einer Imputations-Strategie füllen (z.B. mit dem Mittelwert).

def get_time_window_table(ddf, num_hours_per_window):
    print(datetime.datetime.now().strftime("%H:%M:%S") + " - Erstelle Zeitfenstertabelle...")
    num_mins_per_window = num_hours_per_window * 60
    ddf['zeitfenster'] = (np.floor(ddf['minuten_vor_prozedur'] / num_mins_per_window)).astype(int)

    ddf_zeitfenster_grouped = ddf[['parameterid_effektiv', 'zeitfenster']].groupby(
        by=['parameterid_effektiv', 'zeitfenster']
    ).size().reset_index().compute()
    ddf_zeitfenster_grouped.columns = ['parameterid_effektiv', 'zeitfenster', 'zeitfenster_size']
    ddf_zeitfenster_pivot = ddf_zeitfenster_grouped.pivot(
        index='parameterid_effektiv',
        columns='zeitfenster',
        values='zeitfenster_size'
    ).fillna(0)
    ddf_zeitfenster_pivot = ddf_zeitfenster_pivot.astype(int)

    name_last_column = ddf_zeitfenster_pivot.columns[-1]
    ddf_prozeduren_dedup = ddf.drop_duplicates(subset=['Fallnummer', 'prozedur_datetime']).copy()
    num_prozeduren = len(ddf_prozeduren_dedup)
    print(f"Anzahl eindeutiger Prozeduren: {num_prozeduren}")
    ddf_zeitfenster_pivot['abdeckung'] = ddf_zeitfenster_pivot[0].apply(
        lambda x: round(x * 100 / num_prozeduren, 1)
    )

    return ddf_zeitfenster_pivot

def create_time_window_heatmap(pivot_df, num_hours_per_window):
    plt.figure(figsize=(20, 30))  # Passe die Größe bei Bedarf an
    sns.heatmap(
        pivot_df,
        annot=True,
        linewidth=.5,
        cmap='crest',
        fmt='d',
        annot_kws={'fontsize': 10},
    )

    zeitfenster_labels = [f'{i * num_hours_per_window}-{(i + 1) * num_hours_per_window}h' for i in pivot_df.columns]
    plt.xticks(ticks=np.arange(len(zeitfenster_labels)) + 0.5, labels=zeitfenster_labels, rotation=45, ha='right')
    plt.title('Zeitliche Verteilung der Laborparameter vor der Prozedur', fontsize=16)
    plt.xlabel(f'{num_hours_per_window}h-Fenster vor der Prozedur', fontsize=12)
    plt.ylabel('Laborparameter', fontsize=12)
    plt.tight_layout()  # Stellt sicher, dass alles gut lesbar ist

    output_filename = get_now_label() + "time_window_verteilung_grouped.png"
    plt.savefig(output_filename)
    plt.close()
    print(f"\nGrafik wurde erfolgreich als '{output_filename}' gespeichert.")

def main():
    print(datetime.datetime.now().strftime("%H:%M:%S") + " - Let's go!")
    # 1. Hole Prozeduren
    df_prozeduren = get_labeled_prozeduren_from_file()

    # 1.1 Filtere auf LAE-Kategorie 0 (=LAE ausgeschlossen) und 1 (=LAE nachgewiesen) und confidence >= 0.9
    df_prozeduren_for_training = df_prozeduren[
        ((df_prozeduren['predicted_label'] == "Keine LE (0)")
        | (df_prozeduren['predicted_label'] == "LE vorhanden (1)"))
        & (df_prozeduren['confidence'] >= 0.9)
    ].copy()

    print(f"{len(df_prozeduren_for_training)} von {len(df_prozeduren)} Prozeduren haben ein "
          f"confidence-Wert >= 0.9 und sind in LAE-Kategorie 'Keine LE (0)' oder 'LE vorhanden (1)'.")

    # 2. Hole die Pivottabelle mit Laborwerten
    proz_lab_pivot = get_prozedur_labor_pivot(
        df_prozeduren_for_training,
        labor_window_in_hours=168,
        variant='complete'
    )
    print(datetime.datetime.now().strftime("%H:%M:%S") + " - Schritt 2 (proz_lab_pivot) ist fertig")

    # proz_lab_pivot.to_csv(get_now_label() + "prozedur_labor_pivot.csv", index=False)

    # 3. Hole die Pivottabelle mit Diagnosen
    proz_diagnosen_pivot = get_prozedur_charlson_pivot(df_prozeduren_for_training)
    print(datetime.datetime.now().strftime("%H:%M:%S") + " - Schritt 3 (proz_diagnosen_pivot) ist fertig")

    # 4. Erstelle die finale Tabelle
    # 4.1 Merge Labor auf Prozeduren
    df_proz_mit_labor = pd.merge(
        df_prozeduren_for_training,
        proz_lab_pivot,
        on=['Fallnummer', 'prozedur_datetime'],
        how='inner'
    )

    # 4.2 Merge Diagnosen auf Prozeduren
    df_final_for_training = pd.merge(
        df_proz_mit_labor,
        proz_diagnosen_pivot,
        on=['Fallnummer', 'prozedur_datetime'],
        how='left'
    )
    print(datetime.datetime.now().strftime("%H:%M:%S") + " - Schritt 4 (merges) ist fertig")

    # 5. Fülle alle NaN-Felder in den Charlsonspalten mit False, da dort keine qualifizierenden Diagnosen vorliegen.
    #    Füge außerdem die Spalte 'predicted_label_int' fürs Training hinzu
    for i in range(1, 18):
        df_final_for_training[f'charlson_group_{i}'] = df_final_for_training[
            f'charlson_group_{i}'].fillna(False)
    df_final_for_training['predicted_label_int'] = df_final_for_training['predicted_label'].apply(
        lambda x: 0 if x == 'Keine LE (0)' else 1
    )

    # 6. Filtere auf relevante Spalten

    columns_to_drop = [
        'prozedur_datum',
        'prozedur_zeit',
        'befund_datum',
        'ORGFA',
        'ORGPF',
        'DOKNR',
        'ZBEFALL04B',
        'ZBEFALL04D',
        'DODAT',
        'ERDAT',
        'UPDAT',
        'geburtsdatum',
        'CONTENT',
        'predicted_label',
        'confidence',
        'prozedur_fenster_start',
    ]
    df_final_for_training = df_final_for_training.drop(columns=columns_to_drop)

    print("Shape von ddf_final_for_training:")
    print(df_final_for_training.shape)
    # df_final_for_training.dtypes.to_csv(
    #     get_now_label() + "proz_mit_labor_und_diagnosen_dtypes.csv",
    # )
    df_final_for_training.to_csv(
        get_now_label() + "proz_mit_labor_und_diagnosen_final.csv",
        index=False,
        # columns=relevant_columns,
    )

    print(datetime.datetime.now().strftime("%H:%M:%S") + " - All done!")

if __name__ == "__main__":
    main()