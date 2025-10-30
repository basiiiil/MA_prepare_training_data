"""
Dieses Script soll alle Prozessschritte der Datenbereinigung und -zusammenführung nacheinander
ausführen, sodass am Ende nur dieses Script bedient werden muss.
"""
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dask import dataframe as dd

from config import LABOR_PARAMS_EFFEKTIV, CHARLSON_GROUPS
from functions_diagnosen import get_prozedur_charlson_pivot
from functions_labor import get_prozedur_labor_pivot, get_prozedur_labor_pivot_pandas
from util_functions import concat_csv_files


def get_now_label():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M_")

def get_labeled_prozeduren_from_file():
    df_befunde_labeled = concat_csv_files(
        folder_path='befunde_mit_label',
        csv_dtype=None,
        csv_cols=[
            'Fallnummer',
            'prozedur_datetime',
            'Patientennummer',
            'geschlecht',
            'alter_bei_prozedur',
            'predicted_label',
            'confidence'
        ]
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

def get_prozeduren_for_training():
    df_prozeduren = get_labeled_prozeduren_from_file()
    df_prozeduren_dedup = df_prozeduren.drop_duplicates().copy()

    # 1.1 Filtere auf LAE-Kategorie 0 (=LAE ausgeschlossen) und 1 (=LAE nachgewiesen) und confidence >= 0.9
    df_prozeduren_for_training = df_prozeduren_dedup[
        ((df_prozeduren_dedup['predicted_label'] == "Keine LE (0)")
         | (df_prozeduren_dedup['predicted_label'] == "LE vorhanden (1)"))
        & (df_prozeduren_dedup['confidence'] >= 0.9)
        ].copy()

    print(f"{len(df_prozeduren_for_training)} von {len(df_prozeduren)} Prozeduren haben ein "
          f"confidence-Wert >= 0.9 und sind in LAE-Kategorie 'Keine LE (0)' oder 'LE vorhanden (1)'.")

    return df_prozeduren_for_training

def merge_labor_auf_prozeduren_dask(df_prozeduren):
    # 2. Merge Labor auf Prozeduren
    # 2.1 Hole Laborpivot
    proz_lab_pivot_dask = get_prozedur_labor_pivot(
        df_prozeduren,
        labor_window_in_hours=168,
        variant='complete'
    )
    print(datetime.datetime.now().strftime("%H:%M:%S") + " - proz_lab_pivot wurde erfolgreich geladen.")
    # print(proz_lab_pivot_dask.columns)

    # 2.2 Merge Labor auf Prozeduren
    # 2.2.1 Erstelle Spalte 'Fall_dt'
    df_prozeduren['Fallnummer_str'] = df_prozeduren['Fallnummer'].astype(str)
    df_prozeduren['Fall_dt'] = df_prozeduren['Fallnummer_str'].str.cat(
        others=df_prozeduren['prozedur_datetime'].dt.strftime('%Y-%m-%d_%H:%M:%S'),
        sep='+',
    )

    # 2.2.2 Wandle Prozeduren df in dask um
    df_prozeduren_dask = dd.from_pandas(df_prozeduren)

    # 2.2.3 Setze dtypes für Fall_dt Spalten
    df_prozeduren_dask['Fall_dt'] = df_prozeduren_dask['Fall_dt'].astype(str)
    proz_lab_pivot_dask['Fall_dt'] = proz_lab_pivot_dask['Fall_dt'].astype(str)

    # 2.2.4 Reduziere Spalten von proz_lab_pivot_dask auf Laborspalten  plus Fall_dt
    labor_spalten_mit_falldt = LABOR_PARAMS_EFFEKTIV.copy()
    labor_spalten_mit_falldt.append('Fall_dt')
    proz_lab_pivot_dask_reduced = proz_lab_pivot_dask[labor_spalten_mit_falldt]

    # 2.2.5 Merge Labor auf Prozeduren
    df_proz_mit_labor = dd.merge(
        df_prozeduren_dask,
        proz_lab_pivot_dask_reduced,
        on=['Fall_dt'],
        how='inner'
    )
    print(datetime.datetime.now().strftime("%H:%M:%S") + " - Schritt 2 (merge labor auf proz) ist fertig.")
    # print(df_proz_mit_labor.columns)

    return df_proz_mit_labor

def merge_diagnosen_auf_prozeduren_dask(df_prozeduren):
    # 3. Merge Diagnosen auf Prozeduren
    # 3.1 Hole die Pivottabelle mit Diagnosen
    proz_diagnosen_pivot = get_prozedur_charlson_pivot(df_prozeduren)
    # proz_diagnosen_pivot = get_prozedur_charlson_pivot(df_prozeduren_for_training)
    print(datetime.datetime.now().strftime("%H:%M:%S") + " - proz_diagnosen_pivot wurde erfolgreich geladen.")

    # 3.2.1 Erstelle Spalte 'Fall_dt'
    proz_diagnosen_pivot['Fallnummer_str'] = proz_diagnosen_pivot['Fallnummer'].astype(str)
    proz_diagnosen_pivot['Fall_dt'] = proz_diagnosen_pivot['Fallnummer_str'].str.cat(
        others=proz_diagnosen_pivot['prozedur_datetime'].dt.strftime('%Y-%m-%d_%H:%M:%S'),
        sep='+',
    )
    df_prozeduren['Fallnummer_str'] = df_prozeduren['Fallnummer'].astype(str)
    df_prozeduren['Fall_dt'] = df_prozeduren['Fallnummer_str'].str.cat(
        others=df_prozeduren['prozedur_datetime'].dt.strftime('%Y-%m-%d_%H:%M:%S'),
        sep='+',
    )

    # 3.2.2 Wandle Diagnosen df in dask um

    # 3.2.3 Setze dtypes für Fall_dt Spalten
    df_prozeduren['Fall_dt'] = df_prozeduren['Fall_dt'].astype(str)
    proz_diagnosen_pivot['Fall_dt'] = proz_diagnosen_pivot['Fall_dt'].astype(str)

    # 3.2.4 Reduziere Spalten von proz_diagnosen_pivot auf Charlsonspalten  plus Fall_dt
    charlson_spalten_mit_falldt = CHARLSON_GROUPS.copy()
    charlson_spalten_mit_falldt.append('Fall_dt')
    proz_diagnosen_pivot_reduced = proz_diagnosen_pivot[charlson_spalten_mit_falldt]

    # 3.2.5 Merge Diagnosen auf Prozeduren
    df_proz_mit_diagnosen = dd.merge(
        df_prozeduren,
        proz_diagnosen_pivot_reduced,
        on=['Fall_dt'],
        how='left'
    )
    print(datetime.datetime.now().strftime("%H:%M:%S") + " - Schritt 3 (merge diagnosen auf proz/labor) ist fertig.")
    # print(df_final_for_training.columns)

    return df_proz_mit_diagnosen

def main():
    print(datetime.datetime.now().strftime("%H:%M:%S") + " - Let's go!")
    # 1. Hole Prozeduren
    df_prozeduren = get_prozeduren_for_training()

    # 2. Merge Labor auf Prozeduren
    # 2.1 Lade proz_lab_pivot
    proz_lab_pivot = get_prozedur_labor_pivot_pandas(
        df_prozeduren,
        labor_window_in_hours=168,
        variant='complete'
    )
    print(datetime.datetime.now().strftime("%H:%M:%S") + " - proz_lab_pivot erfolgreich geladen.")
    # print(f"Länge proz_lab_pivot: {len(proz_lab_pivot)}")
    # print(proz_lab_pivot.columns)

    # 2.2 Merge (mittels inner join, weil Prozeduren ohne Laborwerte im Laborfenster nicht beachtet werden)
    proz_mit_labor = pd.merge(
        df_prozeduren,
        proz_lab_pivot,
        on=['Fallnummer', 'prozedur_datetime'],
        how='inner'
    )
    # print(f"Länge proz_mit_labor: {len(proz_mit_labor)}")
    # print(proz_mit_labor.columns)

    # 3. Merge Charlson Gruppen auf Prozeduren-Labor-DF
    # 3.1 Lade proz_diagnosen_pivot
    proz_diagnosen_pivot = get_prozedur_charlson_pivot(df_prozeduren)
    print(datetime.datetime.now().strftime("%H:%M:%S") + " - proz_diagnosen_pivot erfolgreich geladen.")
    # print(f"Länge proz_diagnosen_pivot: {len(proz_diagnosen_pivot)}")

    # 3.2 Merge (mittels left join, weil auch Prozeduren ohne Charlson-Gruppen beachtet werden
    proz_mit_labor_und_diagnosen = pd.merge(
        proz_mit_labor,
        proz_diagnosen_pivot,
        on=['Fallnummer', 'prozedur_datetime'],
        how='left'
    )

    print(datetime.datetime.now().strftime("%H:%M:%S") + " - Finaler Merge erfolgreich.")
    # print(proz_mit_labor_und_diagnosen.columns)
    # print(f"Länge proz_mit_labor_und_diagnosen: {len(proz_mit_labor_und_diagnosen)}")

    # 4. Fülle alle NaN-Felder in den Charlsonspalten mit False, da dort keine qualifizierenden Diagnosen vorliegen.
    #    Füge außerdem die Spalte 'predicted_label_int' fürs Training hinzu
    for i in range(1, 18):
        proz_mit_labor_und_diagnosen[f'charlson_group_{i}'] = proz_mit_labor_und_diagnosen[
            f'charlson_group_{i}'].fillna(False)
    proz_mit_labor_und_diagnosen['predicted_label_int'] = proz_mit_labor_und_diagnosen['predicted_label'].apply(
        lambda x: 0 if x == 'Keine LE (0)' else 1
    )

    # 6. Filtere auf relevante Spalten

    # columns_to_drop = [
    #     'prozedur_datum',
    #     'prozedur_zeit',
    #     'befund_datum',
    #     'ORGFA',
    #     'ORGPF',
    #     'DOKNR',
    #     'ZBEFALL04B',
    #     'ZBEFALL04D',
    #     'DODAT',
    #     'ERDAT',
    #     'UPDAT',
    #     'geburtsdatum',
    #     'CONTENT',
    #     'predicted_label',
    #     'confidence',
    #     'prozedur_fenster_start',
    # ]
    # df_final_for_training = df_final_for_training.drop(columns=columns_to_drop)
    #
    print("Shape von proz_mit_labor_und_diagnosen:")
    print(proz_mit_labor_und_diagnosen.shape)
    # proz_mit_labor_und_diagnosen.dtypes.to_csv(
    #     get_now_label() + "proz_mit_labor_und_diagnosen_dtypes.csv",
    # )
    # proz_mit_labor_und_diagnosen.to_csv(
    #     get_now_label() + "proz_mit_labor_und_diagnosen_final.csv",
    #     index=False,
    # )

    print(datetime.datetime.now().strftime("%H:%M:%S") + " - All done!")

if __name__ == "__main__":
    main()