import datetime
from dask import dataframe as dd
import pandas as pd
import numpy as np


"""
WAS DIESES SCRIPT TUN SOLL:
1. Hinter jeden Fall den Zeitstempel der Prozedur hängen
    Das Mapping basiert ausschließlich auf Fallnummer.

2. Laborwertzeilen filtern: Nur Zeilen mit folgenden Kriterien zulassen:
    1. Parameterbezeichnung ist nicht null
    2. Der Ergebniswert E ist eine Dezimalzahl (kein Text)
        Die Zeichen '<' oder '>' werden einfach entfernt.
    3. Die Parameter-ID ist in der Definitionstabelle enthalten
        
2. Die Spalten 'Ergebniswert', 'Referenzwert unten' und 'Referenzwert oben' in numerische Werte umwandeln.
"""

""" --- KONFIGURATION --- """

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
        # 'Auftragsnummer': np.int_,
        'Fallnummer': np.int_,
        'Probeneingangsdatum': np.str_,
        # 'Ergebnisdatum': np.str_,
        'Parameterbezeichnung': np.str_,
        'Ergebniswert': np.str_,
        # 'Ergebniseinheit': np.str_,
        # 'Ergebniseinheit (UCUM)': np.str_,
        # 'Referenzwert unten': np.str_,
        # 'Referenzwert oben': np.str_,
        'Parameter-ID primär': np.str_,
        # 'LOINC-Code': np.str_,
        # 'Probenart': np.str_,
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
    df_labor_complete = dd.concat([df_labor_original, df_labor_fehlende])
    return df_labor_complete

def get_complete_laborwerte_df_pandas():
    dtype_lib = {
        # 'Auftragsnummer': np.int_,
        'Fallnummer': np.int_,
        'Probeneingangsdatum': np.str_,
        # 'Ergebnisdatum': np.str_,
        'Parameterbezeichnung': np.str_,
        'Ergebniswert': np.str_,
        # 'Ergebniseinheit': np.str_,
        # 'Ergebniseinheit (UCUM)': np.str_,
        # 'Referenzwert unten': np.str_,
        # 'Referenzwert oben': np.str_,
        'Parameter-ID primär': np.str_,
        # 'LOINC-Code': np.str_,
        # 'Probenart': np.str_,
    }

    print(datetime.datetime.now().strftime("%H:%M:%S") + " - Lese die Labor-CSVs...")
    df_col_names = pd.read_csv(
        'fromDIZ/20250929_LAE_Risiko_labor_Spaltennamen_AS.csv',
        delimiter=';',
        header=None,
        encoding='iso-8859-1'
    )
    col_names = df_col_names.head(1).squeeze().tolist()
    df_labor_original = pd.read_csv(
        'fromDIZ/Laborwerte/20250929_LAE_Risiko_laboruntersuchungen_AS.csv',
        dtype=dtype_lib,
        usecols=list(dtype_lib.keys()),
        delimiter=';',
        decimal=',',
        header=None,
        names=col_names,
        encoding='utf_8',
    )
    df_labor_fehlende = pd.read_csv(
        'fromDIZ/Laborwerte/20251007_LAE_Risiko_fehlende_laboruntersuchungen_CW.csv',
        dtype=dtype_lib,
        usecols=list(dtype_lib.keys()),
        delimiter=';',
        decimal=',',
        encoding='utf_8',
    )
    df_labor_complete = pd.concat([df_labor_original, df_labor_fehlende])
    return df_labor_complete

def filter_for_relevant_rows(ddf):
    # Erstelle numerische Spalten
    ddf['ergebniswert_num'] = dd.to_numeric(
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
    print(
        datetime.datetime.now().strftime("%H:%M:%S")
        + " - Korrigiere Parameter-ID und Auftragszeitpunkt für Glukose AccuCheck..."
    )

    # parameterid_effektiv und Bezeichnung für Glucose AccuCheck-Werte setzen
    ddf['parameterid_effektiv'] = ddf['Parameter-ID primär'].mask(
        ddf['Parameter-ID primär'].str.contains(r"O-GLU_Z\.\d{4}", regex=True, na=False),
        'O-GLU_Z.x'
    )
    ddf['parameterbezeichnung_effektiv'] = ddf['Parameterbezeichnung'].mask(
        ddf['Parameter-ID primär'].str.contains(r"O-GLU_Z\.\d{4}", regex=True, na=False),
        'Glukose AccuChek'
    )

    # abnahmezeitpunkt_effektiv für alle Werte setzen
    ddf['Probeneingangsdatum'] = dd.to_datetime(
        ddf['Probeneingangsdatum'],
        format='%Y-%m-%dT%H:%M:%S%z'
    )
    ddf['Probeneingangsdatum'] = ddf['Probeneingangsdatum'].dt.tz_localize(None)
    meta = ('abnahmezeitpunkt_effektiv', ddf['Probeneingangsdatum'].dtype)
    ddf['abnahmezeitpunkt_effektiv'] = ddf.map_partitions(
        calculate_effective_time,
        r"O-GLU_Z\.(\d{4})",
        meta=meta,
    )

    # parameterid_effektiv für Chlorid, Kalium und Natrium aus Serum und Vollblut zusammenführen
    # Chlorid: 'CL_S' und 'O-CL_Z' zu 'CL_serum_oder_bga'
    ddf['parameterid_effektiv'] = ddf['parameterid_effektiv'].mask(
        (ddf['Parameter-ID primär'].str.fullmatch("CL_S") | ddf['Parameter-ID primär'].str.fullmatch("O-CL_Z")),
        'CL_serum_oder_bga'
    )
    # Kalium: 'K_S' und 'O-K_Z' zu 'K_serum_oder_bga'
    ddf['parameterid_effektiv'] = ddf['parameterid_effektiv'].mask(
        (ddf['Parameter-ID primär'].str.fullmatch("K_S") | ddf['Parameter-ID primär'].str.fullmatch("O-K_Z")),
        'K_serum_oder_bga'
    )
    # Natrium: 'NA_S' und 'O-NA_Z' zu 'NA_serum_oder_bga'
    ddf['parameterid_effektiv'] = ddf['parameterid_effektiv'].mask(
        (ddf['Parameter-ID primär'].str.fullmatch("NA_S") | ddf['Parameter-ID primär'].str.fullmatch("O-NA_Z")),
        'NA_serum_oder_bga'
    )

    # parameterbezeichnung_effektiv für Chlorid, Kalium und Natrium aus Serum und Vollblut zusammenführen
    # Chlorid: 'CL_S' und 'O-CL_Z' zu 'Chlorid (Serum oder BGA)'
    ddf['parameterbezeichnung_effektiv'] = ddf['parameterbezeichnung_effektiv'].mask(
        (ddf['Parameter-ID primär'].str.fullmatch("CL_S") | ddf['Parameter-ID primär'].str.fullmatch("O-CL_Z")),
        'Chlorid (Serum oder BGA)'
    )
    # Kalium: 'K_S' und 'O-K_Z' zu 'Kalium (Serum oder BGA)'
    ddf['parameterbezeichnung_effektiv'] = ddf['parameterbezeichnung_effektiv'].mask(
        (ddf['Parameter-ID primär'].str.fullmatch("K_S") | ddf['Parameter-ID primär'].str.fullmatch("O-K_Z")),
        'Kalium (Serum oder BGA)'
    )
    # Natrium: 'NA_S' und 'O-NA_Z' zu 'Natrium (Serum oder BGA)'
    ddf['parameterbezeichnung_effektiv'] = ddf['parameterbezeichnung_effektiv'].mask(
        (ddf['Parameter-ID primär'].str.fullmatch("NA_S") | ddf['Parameter-ID primär'].str.fullmatch("O-NA_Z")),
        'Natrium (Serum oder BGA)'
    )

    return ddf

def normalize_ids_and_timestamps_pandas(df):
    print(
        datetime.datetime.now().strftime("%H:%M:%S")
        + " - Korrigiere Parameter-ID und Auftragszeitpunkt für Glukose AccuCheck..."
    )

    df['parameterid_effektiv'] = df['Parameter-ID primär'].mask(
        df['Parameter-ID primär'].str.contains(r"O-GLU_Z\.\d{4}", regex=True, na=False),
        'O-GLU_Z.x'
    )
    df['parameterbezeichnung_effektiv'] = df['Parameterbezeichnung'].mask(
        df['Parameter-ID primär'].str.contains(r"O-GLU_Z\.\d{4}", regex=True, na=False),
        'Glukose AccuChek'
    )
    df['Probeneingangsdatum'] = pd.to_datetime(
        df['Probeneingangsdatum'],
        format='%Y-%m-%dT%H:%M:%S%z',
        errors='coerce'
    )
    df['Probeneingangsdatum'] = df['Probeneingangsdatum'].dt.tz_localize(None)
    df['abnahmezeitpunkt_effektiv'] = calculate_effective_time(df,r"O-GLU_Z\.(\d{4})")

    return df

def get_labor_ddf():
    ddf_labor = get_complete_laborwerte_ddf()
    ddf_labor_filtered = filter_for_relevant_rows(ddf_labor)
    ddf_labor_normalized = normalize_ids_and_timestamps(ddf_labor_filtered)

    return ddf_labor_normalized