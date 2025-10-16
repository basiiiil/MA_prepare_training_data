"""
Dieses Script soll alle Prozessschritte der Datenbereinigung und -zusammenführung nacheinander
ausführen, sodass am Ende nur dieses Script bedient werden muss.
"""
import pandas as pd
from get_source_dfs import get_stammdaten_inpatients_df, get_prozeduren_df, get_befunde_df
from merge_befunde_and_prozeduren import merge_befunde_and_prozeduren
from add_laborwerte_to_prozeduren import add_laborwerte_to_prozeduren

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
    df_prozeduren_with_labor = add_laborwerte_to_prozeduren(df_prozeduren_final)
    # print(len(df_prozeduren_with_labor))

    # Berechne die Zeitdifferenz zwischen Prozedur und Laborabnahme
    time_delta_seconds = (
            df_prozeduren_with_labor['prozedur_datetime'] - df_prozeduren_with_labor[
        'abnahmezeitpunkt_effektiv']
    ).dt.total_seconds()

    # Rechne es in eine praktischere Einheit um, z.B. Stunden vor der Prozedur
    df_prozeduren_with_labor['stunden_vor_prozedur'] = time_delta_seconds / 3600

    # Analysiere die Verteilung dieser Zeitdifferenz für die häufigsten Parameter
    # Wir können uns z.B. die deskriptive Statistik pro Parameter ansehen
    zeitverteilung_stats = df_prozeduren_with_labor.groupby(
        'parameterid_effektiv'
    )['stunden_vor_prozedur'].describe()

    # Ergebnis berechnen und anzeigen
    print("\nZeitliche Verteilung der Laborparameter (in Stunden vor der Prozedur):")
    print(zeitverteilung_stats.compute())



if __name__ == "__main__":
    main()