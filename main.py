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
    alter_aus_gebdatum = df_prozeduren_final['alter'].fillna(
        (df_prozeduren_final['prozedur_datetime'] - df_prozeduren_final['geburtsdatum']).dt.days
    )
    alter_aus_gebdatum_float = alter_aus_gebdatum / 365.25
    df_prozeduren_final['alter'] = df_prozeduren_final['alter'].fillna(
        alter_aus_gebdatum_float.round().astype(int)
    )
    df_prozeduren_final['altersdekade_bei_prozedur'] = np.ceil(df_stammdaten_inpatients['alter'] / 10)

    return df_prozeduren_final

# def filter_for_latest_results(ddf):


def create_laborwerte_violinplot(df_labor, parameter):
    # Sortiere die Kategorien im DataFrame nach der Häufigkeit für einen sauberen Plot
    df_labor['parameterid_effektiv'] = pd.Categorical(
        df_labor['parameterid_effektiv'], categories=parameter, ordered=True
    )
    df_plot = df_labor.sort_values('parameterid_effektiv')

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

    # Laborwerte zu Prozeduren hinzufügen
    ddf_analyse = add_laborwerte_to_prozeduren(df_prozeduren_final)

    # 1. Berechne die neue Spalte 'stunden_vor_prozedur'
    time_delta_seconds = (
            ddf_analyse['prozedur_datetime'] - ddf_analyse['abnahmezeitpunkt_effektiv']
    ).dt.total_seconds()
    ddf_analyse['stunden_vor_prozedur'] = time_delta_seconds / 3600

    # 2. Berechne Anzahl und Mittelwert für jeden Parameter
    # zeitverteilung_stats = ddf_analyse.groupby('parameterid_effektiv').agg({
    #     'stunden_vor_prozedur': ['mean', 'count']
    # }).compute()

    ddf_analyse_pd = ddf_analyse[
        ['parameterid_effektiv', 'stunden_vor_prozedur']
    ].copy().compute()
    print(f"Anzahl der Laborparameter zu qualifizierten Prozeduren: {len(ddf_analyse_pd)}")
    zeitverteilung_stats = ddf_analyse_pd.groupby(
        by='parameterid_effektiv',
    )[['stunden_vor_prozedur']].agg(['mean', 'count'])

    # 5. Ergebnis berechnen und anzeigen
    zeitverteilung_stats.columns = ['mittlere_stunden_vor_prozedur', 'anzahl']  # Spaltennamen vereinfachen
    zeitverteilung_stats = zeitverteilung_stats.sort_values(by='anzahl', ascending=False)

    parameter = zeitverteilung_stats.index.tolist()

if __name__ == "__main__":
    main()