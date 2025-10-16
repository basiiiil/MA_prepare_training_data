"""
Dieses Script soll alle Prozessschritte der Datenbereinigung und -zusammenführung nacheinander
ausführen, sodass am Ende nur dieses Script bedient werden muss.
"""
import pandas as pd
from get_source_dfs import get_stammdaten_inpatients_df, get_prozeduren_df, get_befunde_df
from merge_befunde_and_prozeduren import merge_befunde_and_prozeduren
from add_laborwerte_to_prozeduren import add_laborwerte_to_prozeduren
import matplotlib.pyplot as plt
import seaborn as sns

LABOR_WINDOW_SIZE = 24 * 7 # Fenster der einzubeziehenden Laborwerte, in Stunden

def main():
    # 1. Daten importieren
    df_stammdaten_inpatients = get_stammdaten_inpatients_df()
    """
    ACHTUNG!
    - Der Befundeimport muss so geändert werden, dass die gelabelten Befunde geladen werden!
    - Um die zu labelnden Befunde zu erhalten, kann get_befunde_df() genutzt werden,
        muss dann aber noch mittels Stammdaten auf stationäre Fälle gefiltert werden.
    - Der Stammdatenimport enthält noch keine Geburtsdaten!
    """
    df_befunde = get_befunde_df()
    df_prozeduren = get_prozeduren_df()

    # 2. Prozedurentabelle erstellen
    # 2.1 Befunde auf stationäre Fälle filtern
    df_befunde_inpatients = df_befunde[df_befunde['Fallnummer'].isin(df_stammdaten_inpatients['Fallnummer'])]
    print(f"{len(df_befunde_inpatients)} von {len(df_befunde)} Befunden sind von stationären Fällen.")

    # 2.2 Befunde und Prozeduren so zusammenbringen, dass für jeden Fall pro Tag nur eine Prozedur existiert.
    df_prozeduren_with_befund = merge_befunde_and_prozeduren(
        df_befunde=df_befunde_inpatients,
        df_prozeduren=df_prozeduren,
    )

    # 2.3 Die Ergebnistabelle mit folgenden Spalten erstellen:
    #   Fallnummer, prozedur_datum, prozedur_zeit, geschlecht, geburtsdatum, lae_kategorie
    #   UNIQUE KEY ist Fallnummer + prozedur_datum
    """ Achtung (Stand 16.10.): Die Stammdaten enthalten noch keine Geburtsdaten! """
    df_prozeduren_inpatients = pd.merge(
        df_prozeduren_with_befund,
        df_stammdaten_inpatients,
        on=['Fallnummer'],
        how='inner',
    )
    # 2.4 Überflüssige Spalten verwerfen
    df_prozeduren_final = df_prozeduren_inpatients[[
        'Fallnummer',
        'prozedur_datum',
        'prozedur_zeit',
        'geschlecht',
        'geburtsdatum',
        # 'lae_kategorie'
    ]].copy()

    print(f"Von {len(df_befunde_inpatients)} Befunden stationärer Fälle "
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

    # Laborwerte zu Prozeduren hinzufügen
    ddf_analyse = add_laborwerte_to_prozeduren(df_prozeduren_final)

    # 1. Berechne die neue Spalte 'stunden_vor_prozedur'
    time_delta_seconds = (
            ddf_analyse['prozedur_datetime'] - ddf_analyse['abnahmezeitpunkt_effektiv']
    ).dt.total_seconds()
    ddf_analyse['stunden_vor_prozedur'] = time_delta_seconds / 3600

    # 2. Berechne Anzahl und Mittelwert für jeden Parameter
    zeitverteilung_stats = ddf_analyse.groupby('parameterid_effektiv').agg({
        'stunden_vor_prozedur': ['mean', 'count']
    }).compute()

    # 5. Ergebnis berechnen und anzeigen
    zeitverteilung_stats.columns = ['mittlere_stunden_vor_prozedur', 'anzahl']  # Spaltennamen vereinfachen
    zeitverteilung_stats = zeitverteilung_stats.sort_values(by='anzahl', ascending=False)
    print("\nStatistik der zeitlichen Verteilung (Top 10):")
    print(zeitverteilung_stats.head(10))

    # Filtere die 20 häufigsten Parameter für eine übersichtliche Darstellung
    top_20_parameter = zeitverteilung_stats.head(20).index.tolist()
    ddf_top_20 = ddf_analyse[ddf_analyse['parameterid_effektiv'].isin(top_20_parameter)]

    # Berechne die Daten für den Plot. Wir brauchen hierfür einen pandas DataFrame.
    df_plot = ddf_top_20[['parameterid_effektiv', 'stunden_vor_prozedur']].compute()

    # Sortiere die Kategorien im DataFrame nach der Häufigkeit für einen sauberen Plot
    df_plot['parameterid_effektiv'] = pd.Categorical(df_plot['parameterid_effektiv'], categories=top_20_parameter,
                                                     ordered=True)
    df_plot = df_plot.sort_values('parameterid_effektiv')

    # Erstelle die Visualisierung
    plt.figure(figsize=(12, 10))  # Passe die Größe bei Bedarf an
    sns.boxplot(
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
    output_filename = "laborparameter_verteilung_boxplot.png"
    plt.savefig(output_filename)
    print(f"\nGrafik wurde erfolgreich als '{output_filename}' gespeichert.")

if __name__ == "__main__":
    main()