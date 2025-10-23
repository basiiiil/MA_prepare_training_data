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

from functions_befunde import get_befunde_df
from functions_diagnosen import get_prozedur_charlson_pivot
from functions_labor import get_prozedur_labor_pivot
from functions_prozeduren import get_prozeduren_df
from functions_stammdaten import get_stammdaten_inpatients_df
from merge_data import merge_befunde_and_prozeduren

def get_now_label():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M_")

def get_labelled_prozeduren(labor_window_in_hours):
    # 1. Daten importieren
    df_stammdaten_inpatients = get_stammdaten_inpatients_df()
    """
    ACHTUNG!
    - Der Befundeimport muss so geändert werden, dass die gelabelten Befunde geladen werden!
    """
    df_befunde = get_befunde_df()
    df_prozeduren = get_prozeduren_df()

    # 2. Prozedurentabelle erstellen
    # 2.1 Befunde und Prozeduren so zusammenbringen, dass für jeden Fall pro Tag nur eine Prozedur existiert.
    df_prozeduren_with_befund = merge_befunde_and_prozeduren(
        df_befunde=df_befunde,
        df_prozeduren=df_prozeduren,
    )

    # 2.2 Die Ergebnistabelle mit folgenden Spalten erstellen:
    #   Fallnummer, prozedur_datum, prozedur_zeit, geschlecht, geburtsdatum, lae_kategorie
    #   UNIQUE KEY ist Fallnummer + prozedur_datum
    df_prozeduren_inpatients = pd.merge(
        df_prozeduren_with_befund,
        df_stammdaten_inpatients,
        on=['Fallnummer'],
        how='inner',
    )

    print(f"Von {len(df_prozeduren)} Prozeduren sind "
          f"{len(df_prozeduren_inpatients)} von stationären Fällen.")

    # 3. Datums- und Zeitspalten in datetime Objekte umwandeln
    df_prozeduren_inpatients['prozedur_datetime'] = pd.to_datetime(
        df_prozeduren_inpatients['prozedur_datum'] + "_" + df_prozeduren_inpatients['prozedur_zeit'],
        format='%Y-%m-%d_%H:%M:%S',
    )
    # 3.1 Neue Spalte für Beginn des Laborzeitfensters definieren
    df_prozeduren_inpatients['prozedur_fenster_start'] = pd.to_datetime(
        df_prozeduren_inpatients['prozedur_datetime'] - pd.Timedelta(hours=labor_window_in_hours),
    )
    # 3.2 Alter für alle fehlenden Fälle berechnen, als Diff zwischen GebDatum und prozedur_datetime
    alter_aus_gebdatum = df_prozeduren_inpatients['alter_bei_prozedur'].fillna(
        (df_prozeduren_inpatients['prozedur_datetime'] - df_prozeduren_inpatients['geburtsdatum']).dt.days
    )
    alter_aus_gebdatum_float = alter_aus_gebdatum / 365.25
    df_prozeduren_inpatients['alter_bei_prozedur'] = df_prozeduren_inpatients['alter_bei_prozedur'].fillna(
        alter_aus_gebdatum_float.round().astype(int)
    )
    df_prozeduren_inpatients['altersdekade_bei_prozedur'] = np.ceil(df_stammdaten_inpatients['alter_bei_prozedur'] / 10)
    # 3.3 Entferne Prozeduren mit Alter < 18 Jahren
    df_prozeduren_no_minors = df_prozeduren_inpatients.query("alter_bei_prozedur >= 18").copy()
    print(f"Davon sind {len(df_prozeduren_no_minors)} mit alter_bei_prozedur >= 18.\n")

    df_prozeduren_final = df_prozeduren_no_minors[
        [
            'Fallnummer',
            'Patientennummer',
            'prozedur_datetime',
            'prozedur_fenster_start',
            'alter_bei_prozedur',
            'geschlecht',
            'altersdekade_bei_prozedur'
        ]
    ]

    return df_prozeduren_final

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
    # 1. Hole Prozeduren, mit Zeitfenster gesamt = 7 Tage * 24h = 168h
    df_prozeduren = get_labelled_prozeduren(168)
    """ ACHTUNG! Der Befundeimport muss so geändert werden, dass die gelabelten Befunde geladen werden! """

    # 2. Hole die Pivottabelle mit Laborwerten
    proz_lab_pivot = get_prozedur_labor_pivot(df_prozeduren)

    # proz_lab_pivot.to_csv(get_now_label() + "prozedur_labor_pivot.csv", index=False)


    # 3. Hole die Pivottabelle mit Diagnosen
    proz_diagnosen_pivot = get_prozedur_charlson_pivot(df_prozeduren)

    # 4. Erstelle die finale Tabelle
    # 4.1 Merge Labor auf Prozeduren
    ddf_proz_mit_labor = pd.merge(
        df_prozeduren,
        proz_lab_pivot,
        on=['Fallnummer', 'prozedur_datetime'],
        how='left',
    ).reset_index()

    # 4.2 Merge Diagnosen auf Prozeduren
    ddf_proz_mit_labor_und_diagnosen = pd.merge(
        ddf_proz_mit_labor,
        proz_lab_pivot,
        on=['Fallnummer', 'prozedur_datetime'],
        how='left',
    ).reset_index()

    print(f"Länge von ddf_proz_mit_labor_und_diagnosen: {len(ddf_proz_mit_labor_und_diagnosen)}")
    print(ddf_proz_mit_labor_und_diagnosen.dtypes)




    # ddf_proz_with_lab_latest = get_latest_lab_values(ddf_proz_with_lab)
    # ddf_proz_with_lab_latest_small = ddf_proz_with_lab_latest[
    #     [
    #         'Fallnummer',
    #         'Patientennummer',
    #         'prozedur_datetime',
    #         'alter_bei_prozedur',
    #         'geschlecht',
    #         'parameterid_effektiv',
    #         'ergebniswert_num',
    #         'minuten_vor_prozedur',
    #     ]
    # ]

    # 4. Erstelle Pivottabelle mit Laborwerten als Spalten
    # ddf_final_pivot = get_final_pivot_table(ddf_proz_with_lab_latest_small)




    # ddf_proz_with_lab_latest_small.to_csv(
    #     get_now_label() + 'prozeduren_with_latest_lab_values.csv',
    #     single_file=True,
    #     index=False,
    # )



    # ddf = dd.read_csv('2025-10-19_laborwerte_filtered.csv')
    # ddf_dedup_cases = ddf.drop_duplicates(subset=['Fallnummer', 'prozedur_datetime']).copy()
    # num_cases = len(ddf_dedup_cases)
    # print(num_cases)
    # zeitfenster_pivot = get_time_window_table(ddf, 8)
    # zeitfenster_pivot = zeitfenster_pivot.reset_index(drop=True).rename_axis(None, axis=1)
    # zeitfenster_pivot['anzahl_einträge'] = zeitfenster_pivot['parameterid_effektiv'].apply(
    #     lambda param: len(ddf[ddf['parameterid_effektiv'] == param])
    # )
    # zeitfenster_pivot['abdeckung'] = np.round((zeitfenster_pivot['anzahl_einträge'] * 100 / num_cases), 1)
    # zeitfenster_labels = ['parameterid_effektiv']
    # zeitfenster_labels.extend([f'{i * 8}-{(i + 1) * 8}h' for i in range(21)])
    # zeitfenster_labels.extend(['anzahl_einträge', 'abdeckung'])
    # zeitfenster_pivot.columns = zeitfenster_labels
    # zeitfenster_pivot.to_csv(
    #     get_now_label() + "laborwerte_pro_zeitfenster_mit_abdeckung.csv",
    # )


if __name__ == "__main__":
    main()