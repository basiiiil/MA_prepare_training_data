import datetime
from dask import dataframe as dd
import pandas as pd
from pandas import CategoricalDtype

from config import LABOR_PARAMS_EFFEKTIV

"""
WAS DIESES SCRIPT TUN SOLL:
1. Hinter jeden Fall den Zeitstempel der Prozedur hängen
    Das Mapping basiert ausschließlich auf Fallnummer.

2. Laborwertzeilen filtern: Nur Zeilen mit folgenden Kriterien zulassen:
    1. Parameterbezeichnung ist nicht null
    2. Der Ergebniswert E ist eine Dezimalzahl (kein Text)
        Die Zeichen '<' oder '>' werden einfach entfernt.
    3. Die Parameter-ID ist in der Definitionstabelle enthalten
        
3. Die Spalte 'Ergebniswert' in numerische Werte umwandeln.
"""

""" --- KONFIGURATION --- """

PARAMS_TO_INCLUDE = [
    'B-MCH_E',
    'B-WBC_E',
    'NA_S',
    'O-NA_Z',
    'B-HCT_E',
    'B-MCV_E',
    'B-HGB_E',
    'B-MCHC_E',
    'B-RBC_E',
    'B-RDW_E',
    'CKDEPI',
    'CRE_S',
    'K_S',
    'O-K_Z',
    # 'PTS_C',
    # 'PT_C',
]

dtype_lib_laborwerte = {
    'Auftragsnummer': 'Int64',
    'Fallnummer': 'Int64',
    'Probeneingangsdatum': 'string',
    'Ergebnisdatum': 'string',
    'Parameterbezeichnung': 'string',
    'Ergebniswert': 'string',
    'Ergebniseinheit': 'string',
    'Ergebniseinheit (UCUM)': 'string',
    'Referenzwert unten': 'string',
    'Referenzwert oben': 'string',
    'Parameter-ID primär': 'string',
    'LOINC-Code': 'string',
    'Probenart': 'string',
}

dtype_lib_befunde = {
    'Fallnummer': 'Int64',
    'prozedur_datum': 'string',
    'start_time': 'string',
}


def get_complete_laborwerte_ddf():
    dtype_lib = {
        # 'Auftragsnummer': int,
        'Fallnummer': int,
        'Probeneingangsdatum': str,
        'Ergebnisdatum': str,
        # 'Parameterbezeichnung': str,
        'Ergebniswert': str,
        # 'Ergebniseinheit': str,
        # 'Ergebniseinheit (UCUM)': str,
        # 'Referenzwert unten': str,
        # 'Referenzwert oben': str,
        'Parameter-ID primär': str,
        # 'LOINC-Code': str,
        # 'Probenart': str,
    }

    dtypes_lis = {
        'FallNr': int,
        'AuftSortDatZeit': str,
        'ErgeDatZeit': str,
        'Ergebnis': str,
        'AnfoCode': str,
    }

    print(datetime.datetime.now().strftime("%H:%M:%S") + " - Lese die Labor-CSVs...")
    df_col_names = dd.read_csv(
        'fromDIZ/20250929_LAE_Risiko_labor_Spaltennamen_AS.csv',
        delimiter=';',
        header=None,
        encoding='iso-8859-1'
    )
    col_names = df_col_names.head(1).squeeze().tolist()
    df_labor_original = dd.read_csv(
        urlpath='fromDIZ/Laborwerte/20250929_LAE_Risiko_laboruntersuchungen_AS.csv',
        dtype=dtype_lib,
        usecols=dtype_lib.keys(),
        delimiter=';',
        decimal=',',
        header=None,
        names=col_names,
        encoding='utf_8',
        blocksize="64MB"
    )
    df_labor_fehlende = dd.read_csv(
        urlpath='fromDIZ/Laborwerte/20251007_LAE_Risiko_fehlende_laboruntersuchungen_CW.csv',
        dtype=dtype_lib,
        usecols=dtype_lib.keys(),
        delimiter=';',
        decimal=',',
        encoding='utf_8',
        blocksize="64MB"
    )
    df_labor_blutgase = dd.read_csv(
        urlpath='fromDIZ/Laborwerte/20251024_LAE_Risiko_laboruntersuchungen_Blutgase_AS.csv',
        dtype=dtype_lib,
        usecols=dtype_lib.keys(),
        delimiter=';',
        decimal=',',
        encoding='utf_8',
        blocksize="64MB"
    )
    df_labor_kalium = dd.read_csv(
        urlpath='fromDIZ/Laborwerte/K_H_2015-2024.csv',
        dtype=dtypes_lis,
        usecols=dtypes_lis.keys(),
        delimiter=';',
        decimal=',',
        encoding='utf_8',
        blocksize="64MB"
    )
    df_labor_accuchek = dd.read_csv(
        urlpath='fromDIZ/Laborwerte/O-GLU_Z.xxxx_2020-2024.csv',
        dtype=dtypes_lis,
        usecols=dtypes_lis.keys(),
        delimiter=';',
        decimal=',',
        encoding='utf_8',
        blocksize="64MB"
    )

    # Bereinige df_labor_blutgase
    df_labor_blutgase['Probeneingangsdatum'] = df_labor_blutgase['Probeneingangsdatum'].fillna(
        df_labor_blutgase['Ergebnisdatum']
    )
    df_labor_blutgase_filtered = df_labor_blutgase[df_labor_blutgase['Parameter-ID primär'] != 'O-PROBENTYP']

    # Bereinige df_labor_kalium
    df_labor_kalium['Fallnummer'] = df_labor_kalium['FallNr']
    df_labor_kalium['Probeneingangsdatum'] = df_labor_kalium['AuftSortDatZeit']
    df_labor_kalium['Ergebnisdatum'] = df_labor_kalium['ErgeDatZeit']
    df_labor_kalium['Ergebniswert'] = df_labor_kalium['Ergebnis']
    df_labor_kalium['Parameter-ID primär'] = df_labor_kalium['AnfoCode']

    # Bereinige df_labor_accuchek
    df_labor_accuchek['Fallnummer'] = df_labor_accuchek['FallNr']
    df_labor_accuchek['Probeneingangsdatum'] = df_labor_accuchek['AuftSortDatZeit']
    df_labor_accuchek['Ergebnisdatum'] = df_labor_accuchek['ErgeDatZeit']
    df_labor_accuchek['Ergebniswert'] = df_labor_accuchek['Ergebnis']
    df_labor_accuchek['Parameter-ID primär'] = df_labor_accuchek['AnfoCode']

    # Filtere auf relevante Spalten
    df_labor_kalium_slim = df_labor_kalium[[
        'Fallnummer',
        'Probeneingangsdatum',
        'Ergebnisdatum',
        'Ergebniswert',
        'Parameter-ID primär'
    ]]
    df_labor_accuchek_slim = df_labor_accuchek[[
        'Fallnummer',
        'Probeneingangsdatum',
        'Ergebnisdatum',
        'Ergebniswert',
        'Parameter-ID primär'
    ]]

    df_labor_complete = dd.concat([
        df_labor_original,
        df_labor_fehlende,
        df_labor_blutgase_filtered,
        df_labor_kalium_slim,
        df_labor_accuchek_slim,
    ])

    df_labor_complete_slim = df_labor_complete.drop(columns=['Ergebnisdatum'])
    df_labor_complete_slim_nona = df_labor_complete_slim.dropna()

    df_labor_complete_dedup = df_labor_complete_slim_nona.drop_duplicates().copy()

    return df_labor_complete_dedup

def filter_for_relevant_rows(ddf, variant):
    '''
    :param ddf: dd.Dataframe
    :param variant: string
    :return: dd.Dataframe
    '''
    # Erstelle numerische Spalten
    ddf['ergebniswert_num'] = dd.to_numeric(
        ddf['Ergebniswert'].str.replace(',', '.').str.replace('<', '').str.replace('>', ''),
        errors='coerce'
    )
    if variant == 'reduced':
        ddf = ddf[ddf['Parameter-ID primär'].isin(PARAMS_TO_INCLUDE)]

    ddf_clean = ddf[
        ddf['ergebniswert_num'].notnull()
        & ddf['Fallnummer'].notnull()
        & ddf['Probeneingangsdatum'].notnull()
    ].copy()

    return ddf_clean

def filter_for_relevant_rows_pandas(ddf):
    # Erstelle numerische Spalten
    ddf['ergebniswert_num'] = pd.to_numeric(
        ddf['Ergebniswert'].str.replace(',', '.').str.replace('<', '').str.replace('>', ''),
        errors='coerce'
    )
    ddf_clean = ddf[
        ddf['Parameterbezeichnung'].notnull()
        & ddf['ergebniswert_num'].notnull()
        & ddf['Fallnummer'].notnull()
        & ddf['Probeneingangsdatum'].notnull()
    ].copy()

    return ddf_clean

def calculate_effective_time(partition_df, pattern):
    """
    Calculates the new 'abnahmezeitpunkt_effektiv' for a single partition.
    """

    # 1. Extract the four digits ('HHMM') as a string Series (NaN otherwise)
    new_time_str = partition_df['Parameter-ID primär'].str.extract(pattern, expand=False)

    # 2. Define the boolean mask (True where a match occurred, False otherwise)
    mask = new_time_str.notnull()

    # The original datetime Series
    original_dt = partition_df['Probeneingangsdatum']

    # 3. Use mask() to create the resulting Series
    # We create a new Series S_repl that has the calculated datetime only for matching rows,
    # and NaT (Not a Time) for non-matching rows.

    if mask.any():
        # Only operate on the matching subset (for efficiency)
        matching_dt = original_dt[mask]
        matching_time_str = new_time_str[mask]

        # Calculate new hour/minute integers
        new_hours = pd.to_numeric(matching_time_str.str[0:2], errors='coerce').astype('Int64')
        new_minutes = pd.to_numeric(matching_time_str.str[2:4], errors='coerce').astype('Int64')

        # Replace the time parts vectorially for the masked subset
        # .dt.to_pydatetime() is used here because Pandas' .dt.replace() does not
        # accept a Series for the hour/minute argument. Using a list comprehension
        # (which is acceptable inside the Pandas partition) is required for this type of replacement.

        # The most robust way: list comprehension on the small partition subset
        new_datetimes = [
            dt.replace(hour=h, minute=m, second=0, microsecond=0)
            for dt, h, m in zip(matching_dt, new_hours, new_minutes)
        ]

        # Put the new datetimes back into a Series, aligned to the original index
        replacement_series = pd.Series(new_datetimes, index=matching_dt.index)

        # Apply the final mask: replace values in original_dt where mask is True
        result_series = original_dt.mask(mask, replacement_series)
    else:
        # If no match in this partition, just return the original datetime Series
        result_series = original_dt

    return result_series

def normalize_ids_and_timestamps(ddf):
    print(datetime.datetime.now().strftime("%H:%M:%S") + " - Korrigiere Parameter-IDs und Abnahmezeitpunkte...")

    # 1. 'abnahmezeitpunkt_effektiv' für alle Werte setzen.
    # 1.1 Probeneingangsdatum zu datetime
    ddf['Probeneingangsdatum'] = dd.to_datetime(
        ddf['Probeneingangsdatum'],
        format='%Y-%m-%dT%H:%M:%S%z',
        utc=True, # True, um mit unterschiedlichen Offsets umgehen zu können
    )
    ddf['Probeneingangsdatum'] = ddf['Probeneingangsdatum'].dt.tz_localize(None)
    # meta = ('abnahmezeitpunkt_effektiv', ddf['Probeneingangsdatum'].dtype)
    # ddf['abnahmezeitpunkt_effektiv'] = ddf.map_partitions(
    #     calculate_effective_time,
    #     r"O-GLU_Z\.(\d{4})",
    #     meta=meta,
    # )

    # 1.2 Umrechnung der Zeiten der AccuChekWerte: Nutze Zeiten aus 'Parameter-ID primär'
    ddf['probeneingang_tag'] = ddf['Probeneingangsdatum'].dt.floor('D')
    # 1.2.1 Extrahiere Zeitstring
    ddf['accuchek_strings'] = ddf['Parameter-ID primär'].where(
        ddf['Parameter-ID primär'].str.contains(r"O-GLU_Z\.\d{4}", regex=True, na=False)
    )
    ddf['accuchek_strings'] = ddf['accuchek_strings'].str.split('.', n=1, expand=True)[1]
    # 1.2.2 Konvertiere Strings zu Stunden und Minuten (Float wegen Fehlwerten)
    ddf['accuchek_hours'] = ddf['accuchek_strings'].str[:2].astype('f8')
    ddf['accuchek_mins'] = ddf['accuchek_strings'].str[2:4].astype('f8')
    ddf['ac_timedelta'] = dd.to_timedelta(
        ddf['accuchek_hours'], unit='h'
    ) + dd.to_timedelta(ddf['accuchek_mins'], unit='m')
    # 1.2.3 Übernehme 'Probeneingangsdatum' als 'abnahmezeitpunkt_effektiv',
    #       außer für AccuCheckwerte: Nutze Datum plus Zeit aus 'Parameter-ID primär'
    ddf['abnahmezeitpunkt_effektiv'] = ddf['Probeneingangsdatum'].mask(
        ddf['Parameter-ID primär'].str.contains(r"O-GLU_Z\.\d{4}", regex=True, na=False),
        ddf['probeneingang_tag'] + ddf['ac_timedelta']
    )



    # 2. 'parameterid_effektiv' für Glukose und Elektrolyte zusammenführen
    # Glucose: 'G-GLU_K', 'GLU_F', 'GLU_S', 'O-GLU_Z', 'O-GLU_Z.xxxx' zusammenführen zu 'x-GLU_x'
    ddf['parameterid_effektiv'] = ddf['Parameter-ID primär'].mask(
        ddf['Parameter-ID primär'].str.contains(r"GLU_", regex=True, na=False),
        'x-GLU_x'
    )

    # Chlorid: 'CL_S' und 'O-CL_Z' zu 'CL_serum_oder_bga'
    ddf['parameterid_effektiv'] = ddf['parameterid_effektiv'].mask(
        (
                ddf['Parameter-ID primär'].str.fullmatch("CL_S")
                | ddf['Parameter-ID primär'].str.fullmatch("O-CL_Z")
        ),
        'x-CL_x'
    )
    # Kalium: 'K_S' und 'O-K_Z' und 'K_H' zu 'K_serum_oder_bga'
    ddf['parameterid_effektiv'] = ddf['parameterid_effektiv'].mask(
        (
                ddf['Parameter-ID primär'].str.fullmatch("K_S")
                | ddf['Parameter-ID primär'].str.fullmatch("O-K_Z")
                | ddf['Parameter-ID primär'].str.fullmatch("K_H")
        ),
        'x-K_x'
    )
    # Natrium: 'NA_S' und 'O-NA_Z' zu 'NA_serum_oder_bga'
    ddf['parameterid_effektiv'] = ddf['parameterid_effektiv'].mask(
        (
                ddf['Parameter-ID primär'].str.fullmatch("NA_S")
                | ddf['Parameter-ID primär'].str.fullmatch("O-NA_Z")
        ),
        'x-NA_x'
    )

    # 3. 'parameterbezeichnung_effektiv' für Glukose und Elektrolyte zusammenführen
    # ddf['parameterbezeichnung_effektiv'] = ddf['Parameterbezeichnung'].mask(
    #     ddf['Parameter-ID primär'].str.contains(r"GLU_", regex=True, na=False),
    #     'Glukose AccuChek'
    # )
    # Chlorid: 'CL_S' und 'O-CL_Z' zu 'Chlorid (Serum oder BGA)'
    # ddf['parameterbezeichnung_effektiv'] = ddf['parameterbezeichnung_effektiv'].mask(
    #     (ddf['Parameter-ID primär'].str.fullmatch("CL_S") | ddf['Parameter-ID primär'].str.fullmatch("O-CL_Z")),
    #     'Chlorid (Serum oder BGA)'
    # )
    # # Kalium: 'K_S' und 'O-K_Z' zu 'Kalium (Serum oder BGA)'
    # ddf['parameterbezeichnung_effektiv'] = ddf['parameterbezeichnung_effektiv'].mask(
    #     (ddf['Parameter-ID primär'].str.fullmatch("K_S") | ddf['Parameter-ID primär'].str.fullmatch("O-K_Z")),
    #     'Kalium (Serum oder BGA)'
    # )
    # # Natrium: 'NA_S' und 'O-NA_Z' zu 'Natrium (Serum oder BGA)'
    # ddf['parameterbezeichnung_effektiv'] = ddf['parameterbezeichnung_effektiv'].mask(
    #     (ddf['Parameter-ID primär'].str.fullmatch("NA_S") | ddf['Parameter-ID primär'].str.fullmatch("O-NA_Z")),
    #     'Natrium (Serum oder BGA)'
    # )

    ddf_slim = ddf[['Fallnummer', 'parameterid_effektiv', 'abnahmezeitpunkt_effektiv', 'ergebniswert_num']]

    ddf_dedup = ddf_slim.drop_duplicates().copy()

    print(
        datetime.datetime.now().strftime("%H:%M:%S")
        + " - Parameter-IDs und Abnahmezeitpunkte erfolgreich korrigiert."
    )

    return ddf_dedup

def get_labor_ddf(variant):
    '''
    :param variant: string
    :return: dd.Dataframe
    '''
    ddf_labor = get_complete_laborwerte_ddf()
    ddf_labor_filtered = filter_for_relevant_rows(ddf_labor, variant)
    ddf_labor_normalized = normalize_ids_and_timestamps(ddf_labor_filtered)

    return ddf_labor_normalized

def get_latest_lab_values(ddf_labor):
    """
    Filtert ein Dask DataFrame, um nur den letzten Laborwert pro Fallnummer
    und Parameter-ID zu behalten.

    Args:
        ddf_labor (dd.DataFrame): Das DataFrame mit potenziell mehreren Laborwerten
                            pro Fall und Parameter. Es muss die Spalten 'Fallnummer',
                            'parameterid_effektiv' und 'abnahmezeitpunkt_effektiv'
                            enthalten.

    Returns:
        dd.DataFrame: Ein gefiltertes DataFrame, das nur noch den jeweils
                      letzten Laborwert enthält.
    """
    print(
        datetime.datetime.now().strftime("%H:%M:%S")
        + " - Filtere nach dem letzten Laborwert pro Prozedur und Parameter......"
    )
    # Berechne die neue Spalte 'minuten_vor_prozedur'
    time_delta_seconds = (
            ddf_labor['prozedur_datetime'] - ddf_labor['abnahmezeitpunkt_effektiv']
    ).dt.total_seconds()
    ddf_labor['minuten_vor_prozedur'] = time_delta_seconds / 60

    # Sortiere das DataFrame so, dass der neueste Wert für jede Gruppe oben steht.
    # Dies ist eine vorbereitende Operation für drop_duplicates.
    ddf_sorted = ddf_labor.sort_values('minuten_vor_prozedur')

    # Behalte nur den ersten Eintrag pro Gruppe ('Fallnummer' und 'parameterid_effektiv').
    # Da wir absteigend sortiert haben, ist dies automatisch der letzte/neueste Wert.
    ddf_latest = ddf_sorted.drop_duplicates(subset=[
        'Fallnummer', 'prozedur_datetime', 'parameterid_effektiv'
    ]).copy()

    print(datetime.datetime.now().strftime("%H:%M:%S") + " - Filterung abgeschlossen.")
    return ddf_latest

def get_prozedur_labor_pivot_pandas(df_prozeduren, labor_window_in_hours, variant):
    '''
    :param df_prozeduren: pd.Dataframe
    :param labor_window_in_hours: int
    :param variant: 'reduced' or 'complete'
    :return: pd.Dataframe
    '''

    # Neue Spalte für Beginn des Laborzeitfensters definieren
    df_prozeduren['prozedur_fenster_start'] = pd.to_datetime(
        df_prozeduren['prozedur_datetime'] - pd.Timedelta(hours=labor_window_in_hours),
    )

    ddf_labor = get_labor_ddf(variant=variant)

    # 1. Merge Labordaten auf die Prozeduren.
    #   Dabei fliegen alle Labordaten raus, deren Fallnummer nicht in der Prozedurentabelle vorkommt.
    #   Laborwerte werden immer dann zugewiesen, wenn ihr 'abnahmezeitpunkt_effektiv' innerhalb des
    #   Fensters liegt (also nach 'prozedur_fenster_start', aber vor 'prozedur_datetime'.
    #   Pro Prozedur wird es dadurch für jeden Laborwert eine Zeile geben.
    ddf_labor['Fallnummer'] = ddf_labor['Fallnummer'].astype('int32')
    df_prozeduren['Fallnummer'] = df_prozeduren['Fallnummer'].astype('int32')
    ddf_merged = dd.merge(
        df_prozeduren,
        ddf_labor,
        on=['Fallnummer'],
        how='inner',
    ).reset_index(drop=True)

    # 2. Filtere auf Laborwerte, die innerhalb des Laborfensters liegen
    query_str = 'prozedur_fenster_start <= abnahmezeitpunkt_effektiv <= prozedur_datetime'
    ddf_labor_filtered = ddf_merged.query(query_str)

    # 3. Filtere auf die jeweils letzten Werte je Parameter je Prozedur
    ddf_labor_latest = get_latest_lab_values(ddf_labor_filtered)

    # 4. Minimiere die Größe des Dataframes
    ddf_labor_latest_small = ddf_labor_latest[[
        'Fallnummer',
        'prozedur_datetime',
        'parameterid_effektiv',
        'ergebniswert_num'
    ]]
    # ddf_labor_latest_small['Fallnummer'] = ddf_labor_latest_small['Fallnummer'].astype('int32')
    # ddf_labor_latest_small['ergebniswert_num'] = ddf_labor_latest_small['ergebniswert_num'].astype('float32')
    # laborparam_categories = CategoricalDtype(LABOR_PARAMS_EFFEKTIV)
    # ddf_labor_latest_small = ddf_labor_latest_small['parameterid_effektiv'].astype(laborparam_categories)
    print(datetime.datetime.now().strftime("%H:%M:%S") + " - Wandle df_labor_latest in pandas df um...")
    df_labor_latest_pandas = ddf_labor_latest_small.compute()
    print(datetime.datetime.now().strftime("%H:%M:%S") + " - Umwandlung abgeschlossen.")
    # print(f"Länge df_labor_latest_pandas: {len(df_labor_latest_pandas)}")
    # num_proz_df_labor_latest_pandas = df_labor_latest_pandas.drop_duplicates(
    #     subset=['Fallnummer', 'prozedur_datetime']
    # ).copy()
    # print(f"Länge num_proz_df_labor_latest_pandas: {len(num_proz_df_labor_latest_pandas)}")


    # 5. Erstelle Pivottabelle
    #    Wir nutzen set_index().unstack() statt pivot_table(),
    #    um den "ArrayMemoryError" (kartesisches Produkt) zu vermeiden.
    # 5.1. Setze ALLE drei Spalten als Index
    indexed_pandas_df = df_labor_latest_pandas.set_index(
        ['Fallnummer', 'prozedur_datetime', 'parameterid_effektiv']
    )

    # 5.2. Wähle die Wertespalte (jetzt eine Series)
    value_series = indexed_pandas_df['ergebniswert_num']
    # print(f"Länge value_series: {len(value_series)}")

    # 5.3. "Entstapel" die 'parameterid_effektiv'-Ebene zu Spalten
    #     Dies erstellt nur Zeilen für existierende (Fall, Zeit)-Paare.
    pivot_df = value_series.unstack(level='parameterid_effektiv')

    # 5.4. Hol den Index (Fallnummer, prozedur_datetime) als Spalten zurück
    pivot_df = pivot_df.reset_index()

    # 6. Nur für variant='reduced': Wirf alle Zeilen (also Prozeduren) raus, die Fehlwerte in den Laborparametern haben
    # Erklärung: Der 'complete' Datensatz darf keine Fehlwerte haben
    if variant == 'reduced':
        pivot_df_nona = pivot_df.dropna()
        num_proz = len(pivot_df)
        num_proz_nona = len(pivot_df_nona)
        print(f"Von {num_proz} Prozeduren mit mind. einem Laborwert sind {num_proz_nona} "
              f"({round(num_proz_nona *100 / num_proz, 1)}%) vollständig "
              f"für den 'Complete'-Datensatz mit reduzierten Parametern.")
        return pivot_df_nona
    else:
        return pivot_df

def get_prozedur_labor_pivot(df_prozeduren, labor_window_in_hours, variant):
    '''
    :param df_prozeduren: pd.Dataframe
    :param labor_window_in_hours: int
    :param variant: 'reduced' or 'complete'
    :return: pd.Dataframe
    '''

    # Neue Spalte für Beginn des Laborzeitfensters definieren
    df_prozeduren['prozedur_fenster_start'] = pd.to_datetime(
        df_prozeduren['prozedur_datetime'] - pd.Timedelta(hours=labor_window_in_hours),
    )

    ddf_labor = get_labor_ddf(variant=variant)
    ddf_prozeduren = dd.from_pandas(df_prozeduren)

    # 1. Merge Labordaten auf die Prozeduren.
    #   Dabei fliegen alle Labordaten raus, deren Fallnummer nicht in der Prozedurentabelle vorkommt.
    #   Laborwerte werden immer dann zugewiesen, wenn ihr 'abnahmezeitpunkt_effektiv' innerhalb des
    #   Fensters liegt (also nach 'prozedur_fenster_start', aber vor 'prozedur_datetime'.
    #   Pro Prozedur wird es dadurch für jeden Laborwert eine Zeile geben.
    ddf_labor['Fallnummer'] = ddf_labor['Fallnummer'].astype('int32')
    ddf_prozeduren['Fallnummer'] = ddf_prozeduren['Fallnummer'].astype('int32')
    ddf_merged = dd.merge(
        ddf_prozeduren,
        ddf_labor,
        on=['Fallnummer'],
        how='inner',
    ).reset_index(drop=True)

    # 2. Filtere auf Laborwerte, die innerhalb des Laborfensters liegen
    query_str = 'prozedur_fenster_start <= abnahmezeitpunkt_effektiv <= prozedur_datetime'
    ddf_labor_filtered = ddf_merged.query(query_str)

    # 3. Filtere auf die jeweils letzten Werte je Parameter je Prozedur
    ddf_labor_latest = get_latest_lab_values(ddf_labor_filtered)

    ddf_labor_latest['Fallnummer_str'] = ddf_labor_latest['Fallnummer'].astype(str)
    ddf_labor_latest['Fall_dt'] = ddf_labor_latest['Fallnummer_str'].str.cat(
        others=ddf_labor_latest['prozedur_datetime'].dt.strftime('%Y-%m-%d_%H:%M:%S'),
        sep='+',
    )

    # 4. Minimiere die Größe des Dataframes
    print("4. Minimiere die Größe des Dataframes...")
    ddf_labor_latest_small = ddf_labor_latest[[
        'Fall_dt',
        # 'Fallnummer',
        # 'prozedur_datetime',
        'parameterid_effektiv',
        'ergebniswert_num'
    ]]
    # print(f"length of ddf_labor_latest_small: {len(ddf_labor_latest_small)}")
    # print(f"Num unique Fall_dt: {ddf_labor_latest_small['Fall_dt'].nunique().compute()}")

    param_categories = CategoricalDtype(LABOR_PARAMS_EFFEKTIV)

    ddf_labor_latest_small['parameterid_effektiv'] = ddf_labor_latest_small['parameterid_effektiv'].astype(
        param_categories
    )

    ddf_prozeduren['Fallnummer_str'] = ddf_prozeduren['Fallnummer'].astype(str)
    ddf_prozeduren['Fall_dt'] = ddf_prozeduren['Fallnummer_str'].str.cat(
        others=ddf_prozeduren['prozedur_datetime'].dt.strftime('%Y-%m-%d_%H:%M:%S'),
        sep='+',
    )

    print(datetime.datetime.now().strftime("%H:%M:%S") + " - Erstelle Pivottabelle...")
    pivot_ddf = dd.pivot_table(
        df=ddf_labor_latest_small,
        index='Fall_dt',
        columns='parameterid_effektiv',
        values='ergebniswert_num',
        aggfunc='first',
    ).reset_index()

    ddf_proz_with_lab = dd.merge(
        ddf_prozeduren,
        pivot_ddf,
        on=['Fall_dt'],
        how='left'
    ).reset_index(drop=True)


    return ddf_proz_with_lab