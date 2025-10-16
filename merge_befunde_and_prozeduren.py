import numpy as np
import pandas as pd

'''
Zuletzt bearbeitet am 15.10.
Funktion dieses Scripts:
1. Filtert Prozedurenliste nach Fällen, die an einem Kalendertag genau eine 3-222-Prozedur hatten.
2. Vereinigt Befunde und Prozeduren so, dass alle Befunde und Prozeduren ohne Korrelation in der 
    jeweils anderen Tabelle rausfliegen (inner join).
Das Ergebnis ist eine Tabelle mit Befunden, denen jeweils genau eine Uhrzeit zugeordnet ist.
Dabei exisitiert jede Kombi aus Fallnummer und Prozedurdatum genau einmal.  
'''

def merge_befunde_and_prozeduren(df_befunde, df_prozeduren):
    # 1. Prozeduren nach Fall und Tag gruppieren
    proz_per_day_with_time = df_prozeduren.groupby(
        by=['Fallnummer', 'prozedur_datum'], as_index=False, dropna=False
    )[['prozedur_zeit']].agg(
        prozedur_zeit=('prozedur_zeit', lambda x: np.nan if len(x) > 1 else x),
        num_start_times=('prozedur_zeit', 'count'),
    )

    # 2. Nur Prozeduren erhalten, zu deren Tag und Fall keine weitere Prozedur existiert
    proz_with_unique_case_date = proz_per_day_with_time[proz_per_day_with_time['num_start_times'] == 1]

    # 3. Befunde und Prozeduren anhand Fallnummer und Prozedurdatum mergen.
    # Befunde und Prozeduren ohne Entsprechung in der jeweils anderen Tabelle werden ignoriert.
    df_befunde_proz_merged = pd.merge(
        df_befunde, proz_with_unique_case_date,
        on=['Fallnummer', 'prozedur_datum'],
        how='inner'
    )
    print(f"{len(df_befunde_proz_merged)} Befunde konnten eindeutig einer Prozedur zugeordnet werden.")

    # 5. Zur Sicherheit erneut auf Duplikate prüfen. Falls solche existieren, wird eine Exception ausgelöst.
    df_befunde_dedup_grouped = df_befunde_proz_merged.groupby(
        by=['Fallnummer', 'prozedur_datum'], as_index=False
    ).size()
    if len(df_befunde_dedup_grouped[df_befunde_dedup_grouped['size'] != 1]) > 0:
        print(df_befunde_dedup_grouped.value_counts('size', dropna=False))
        raise Exception("Es gibt noch Tage, an denen mehrere Befunde pro Fall existieren.")
    else:
        print("\nFür jeden Falltag existiert genau 1 Befund.")
        return df_befunde_proz_merged

