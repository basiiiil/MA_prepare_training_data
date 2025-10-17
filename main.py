"""
Dieses Script soll alle Prozessschritte der Datenbereinigung und -zusammenführung nacheinander
ausführen, sodass am Ende nur dieses Script bedient werden muss.
"""
import pandas as pd
import numpy as np
from get_source_dfs import get_stammdaten_inpatients_df, get_prozeduren_df, get_befunde_df
import matplotlib.pyplot as plt
import seaborn as sns

from merge_data import add_laborwerte_to_prozeduren, merge_befunde_and_prozeduren

LABOR_WINDOW_SIZE = 24 * 7 # Fenster der einzubeziehenden Laborwerte, in Stunden

def get_labelled_prozeduren():
    # 1. Daten importieren
    df_stammdaten_inpatients = get_stammdaten_inpatients_df()
    """
    ACHTUNG!
    - Der Befundeimport muss so geändert werden, dass die gelabelten Befunde geladen werden!
    - Um die zu labelnden Befunde zu erhalten, kann get_befunde_df() genutzt werden,
        muss dann aber noch mittels Stammdaten auf stationäre Fälle gefiltert werden.
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
    # 2.3 Überflüssige Spalten verwerfen
    df_prozeduren_final = df_prozeduren_inpatients[[
        'Fallnummer',
        'prozedur_datum',
        'prozedur_zeit',
        'geschlecht',
        'geburtsdatum',
        'alter',
        # 'lae_kategorie'
    ]].copy()
    print(f"Von {len(df_befunde)} Befunden stationärer Fälle "
          f"wurden {len(df_prozeduren_final)} in den Stammdaten gefunden.")

    # 3. Datums- und Zeitspalten in datetime Objekte umwandeln
    df_prozeduren_final['prozedur_datetime'] = pd.to_datetime(
        df_prozeduren_final['prozedur_datum'] + "_" + df_prozeduren_final['prozedur_zeit'],
        format='%Y-%m-%d_%H:%M:%S',
    )
    # 3.1 Neue Spalte für Beginn des Laborzeitfensters definieren
    df_prozeduren_final['prozedur_fenster_start'] = pd.to_datetime(
        df_prozeduren_final['prozedur_datetime'] - pd.Timedelta(hours=LABOR_WINDOW_SIZE),
    )
    # 3.2 Alter für alle fehlenden Fälle berechnen, als Diff zwischen GebDatum und prozedur_datetime
    alter_aus_gebdatum = df_prozeduren_final['alter'].fillna(
        (df_prozeduren_final['prozedur_datetime'] - df_prozeduren_final['geburtsdatum']).dt.days
    )
    alter_aus_gebdatum_float = alter_aus_gebdatum / 365.25
    df_prozeduren_final['alter'] = df_prozeduren_final['alter'].fillna(
        alter_aus_gebdatum_float.round().astype(int)
    )
    df_prozeduren_final['altersdekade_bei_prozedur'] = np.ceil(df_stammdaten_inpatients['alter'] / 10)

    print(df_prozeduren_final['alter'].isnull().sum())
    df_prozeduren_final_filtered = df_prozeduren_final[
        [
            'Fallnummer',
            'prozedur_datetime',
            'prozedur_fenster_start',
            'alter',
            'geschlecht',
            'altersdekade_bei_prozedur'
        ]
    ]

    return df_prozeduren_final

def get_latest_lab_values(ddf):
    """
    Filtert ein Dask DataFrame, um nur den letzten Laborwert pro Fallnummer
    und Parameter-ID zu behalten.

    Args:
        ddf (dd.DataFrame): Das DataFrame mit potenziell mehreren Laborwerten
                            pro Fall und Parameter. Es muss die Spalten 'Fallnummer',
                            'parameterid_effektiv' und 'abnahmezeitpunkt_effektiv'
                            enthalten.

    Returns:
        dd.DataFrame: Ein gefiltertes DataFrame, das nur noch den jeweils
                      letzten Laborwert enthält.
    """
    print("Filtere nach dem letzten Laborwert pro Fall und Parameter...")
    # Berechne die neue Spalte 'minuten_vor_prozedur'
    time_delta_seconds = (
            ddf['prozedur_datetime'] - ddf['abnahmezeitpunkt_effektiv']
    ).dt.total_seconds()
    ddf['minuten_vor_prozedur'] = time_delta_seconds / 60

    # Sortiere das DataFrame so, dass der neueste Wert für jede Gruppe oben steht.
    # Dies ist eine vorbereitende Operation für drop_duplicates.
    # persist() kann hier die Performance verbessern, indem das sortierte
    # Zwischenergebnis im Speicher gehalten wird.
    ddf_sorted = ddf.sort_values('minuten_vor_prozedur').persist()

    # Behalte nur den ersten Eintrag pro Gruppe ('Fallnummer' und 'parameterid_effektiv').
    # Da wir absteigend sortiert haben, ist dies automatisch der letzte/neueste Wert.
    ddf_latest = ddf_sorted.drop_duplicates(subset=[
        'Fallnummer', 'prozedur_datetime', 'parameterid_effektiv'
    ]).copy()

    print("Filterung abgeschlossen.")
    return ddf_latest

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
    print(f"\nGrafik wurde erfolgreich als '{output_filename}' gespeichert.")

def get_final_pivot_table(ddf):
    # Beispiel: PIVOT-Tabelle erstellen für das Machine-Learning-Modell
    # Wir wollen eine Tabelle, in der jede Zeile ein Fall ist und jede Spalte ein Laborparameter.
    # Der Wert in der Zelle ist der letzte bekannte 'ergebniswert_num'.

    # 1. Entferne unnötige Spalten, um den Pivot zu beschleunigen
    ddf_pivot_input = ddf[['Fallnummer', 'prozedur_datetime', 'parameterid_effektiv', 'ergebniswert_num']]

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

def get_time_window_table(ddf):
    # teile Minuten vor Prozedur durch 480, um das 8h-Zeitfenster zu berechnen
    ddf['8h_zeitfenster'] = np.floor(ddf['minuten_vor_prozedur'] / 480)

    ddf_zeitfenster_grouped = ddf[['parameterid_effektiv', '8h_zeitfenster']].groupby(
        by=['parameterid_effektiv'],
    )[['8h_zeitfenster']].size().compute()
    ddf_zeitfenster_grouped.to_csv(
        '2025-10-17_zeitreihe.csv',
        index=False,
    )
    """ ERST PIVOT TABLE!!! """



def main():
    # 1. Hole nur Prozeduren, die
    #   - einen eindeutig zugeordneten Befund haben und
    #   - zu stationären Fällen gehören.
    df_prozeduren_final = get_labelled_prozeduren()
    """
    ACHTUNG!
    - Der Befundeimport muss so geändert werden, dass die gelabelten Befunde geladen werden!
    - Um die zu labelnden Befunde zu erhalten, kann get_befunde_df() genutzt werden,
        muss dann aber noch mittels Stammdaten auf stationäre Fälle gefiltert werden.
    """

    # Füge Laborwerte zu Prozeduren hinzu
    ddf_prozeduren_mit_labor = add_laborwerte_to_prozeduren(df_prozeduren_final)

    ddf_filtered = get_latest_lab_values(ddf_prozeduren_mit_labor)
    get_time_window_table(ddf_filtered)


if __name__ == "__main__":
    main()