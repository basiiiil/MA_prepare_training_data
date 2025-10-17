import datetime
from dask import dataframe as dd
import pandas as pd
# import numpy as np

from util_functions import get_complete_laborwerte_ddf

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

    ddf['parameterid_effektiv'] = ddf['Parameter-ID primär'].mask(
        ddf['Parameter-ID primär'].str.contains(r"O-GLU_Z\.\d{4}", regex=True, na=False),
        'O-GLU_Z.x'
    )
    ddf['parameterbezeichnung_effektiv'] = ddf['Parameterbezeichnung'].mask(
        ddf['Parameter-ID primär'].str.contains(r"O-GLU_Z\.\d{4}", regex=True, na=False),
        'Glukose AccuChek'
    )
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

def main():
    ddf_labor = get_complete_laborwerte_ddf()

    ddf_labor_filtered = filter_for_relevant_rows(ddf_labor)
    ddf_labor_normalized = normalize_ids_and_timestamps(ddf_labor_filtered)

    ddf_labor_normalized.head(10000).to_csv(
        'Outputs/2025-10-15_laborwerte_normalized.csv',
        # single_file=True,
        index=False,
    )

if __name__ == '__main__':
    main()