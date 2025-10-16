import numpy as np
import pandas as pd

from befunde_merge_and_filter import dedup_befunde, get_unique_befunde_per_case_day, get_befunde_from_files


def get_stammdaten_inpatients_df():
    """
    :return: pandas dataframe
    """
    df_source = pd.read_csv(
        'fromDIZ/Stammdaten/24.07.2025_Patientenliste_3-222_Stammdaten_sf.csv',
        dtype={
            'Fallnummer': 'Int64',
            'Patientennummer': 'Int64',
            'Alter': 'Int64',
            'Aufnahmedatum': str,
        },
        usecols=[
            'Fallnummer',
            'Patientennummer',
            'Geschlecht',
            'Alter',
            'Bewegung Behandlungsart',
            'Aufnahmedatum'
        ]
    )

    df_source_nachgereicht = pd.read_csv(
        'fromDIZ/Stammdaten/Stammdaten.csv',
        dtype={
            'Patient': 'Int64',
            'Fall': 'Int64',
            'GebDatum': str
        },
        usecols=[
            'Fall',
            'Patient',
            'G',
            'GebDatum',
            'Fa'
        ]
    )

    df_source['behandlungsart'] = df_source['Bewegung Behandlungsart']
    df_source['Alter'] = pd.to_timedelta(df_source['Alter'] * 365.25, unit='days')
    df_source['Aufnahmedatum'] = pd.to_datetime(df_source['Aufnahmedatum'], format='%Y-%m-%d')
    df_source['geburtsdatum'] = df_source['Aufnahmedatum'] - df_source['Alter']
    df_source['geschlecht'] = df_source['Geschlecht']

    df_source_nachgereicht['Fallnummer'] = df_source_nachgereicht['Fall']
    df_source_nachgereicht['Patientennummer'] = df_source_nachgereicht['Patient']
    df_source_nachgereicht['geschlecht'] = df_source_nachgereicht['G']
    df_source_nachgereicht['behandlungsart'] = df_source_nachgereicht['Fa']
    df_source_nachgereicht['geburtsdatum'] = pd.to_datetime(df_source_nachgereicht['GebDatum'], format='%Y-%m-%d')

    # 3. Merge all Fallnummern and drop duplicates
    df_stammdaten_all = pd.concat(
        [
            df_source[['Fallnummer', 'Patientennummer', 'geschlecht', 'geburtsdatum', 'behandlungsart']],
            df_source_nachgereicht[['Fallnummer', 'Patientennummer', 'geschlecht', 'geburtsdatum', 'behandlungsart']],
        ]
    )
    df_stammdaten_all_inpatients = df_stammdaten_all[
        df_stammdaten_all['behandlungsart'] == 'Stationär'
    ].reset_index(drop=True)
    df_stammdaten_dedup = df_stammdaten_all_inpatients.drop_duplicates().copy()

    return df_stammdaten_dedup

def get_befunde_df():
    df_source = get_befunde_from_files()

    df_befunde_dedup = dedup_befunde(df_source)

    df_befunde_unique_case_day = get_unique_befunde_per_case_day(df_befunde_dedup)
    print(f"\nEs existieren {len(df_befunde_unique_case_day)} Befunde, an deren Prozedurdatum "
          f"zu diesem Fall kein anderer Befund vorliegt. Somit ist eine eindeutige Zuordung "
          f"zu einer Prozedur möglich.")

    return df_befunde_unique_case_day

def get_prozeduren_df():
    """
    :return: pandas dataframe
    """
    df_source = pd.read_csv(
        'fromDIZ/Prozeduren/25.07.2025_Prozeduren_sf.csv',
        dtype={
            'Fall': 'Int64',
            'OP-Code Beg.': str,
            'BegZeit': str
        },
        usecols=[
            'Fall',
            'OP-Code Beg.',
            'BegZeit',
            'OP-Code'
        ]
    )

    df_source_nachgereicht = pd.read_csv(
        'fromDIZ/Prozeduren/fehlende Prozeduren.csv',
        dtype={
            'Fallnummer': 'Int64',
            'Prozedurzeit': str
        },
        usecols=[
            'Fallnummer',
            'Prozedurdatum',
            'Prozedurzeit',
            'PrzCode'
        ]
    )

    # 1. Add column 'Fallnummer' to each dataframe
    df_source['Fallnummer'] = df_source['Fall']
    df_source['prozedur_datum'] = df_source['OP-Code Beg.']
    df_source['prozedur_zeit'] = df_source['BegZeit'].replace(
        "00:00:00", np.nan
    )

    # 1b 'fehlende Prozeduren.csv'
    df_source_nachgereicht['prozedur_datum'] = df_source_nachgereicht['Prozedurdatum']
    df_source_nachgereicht['prozedur_zeit'] = df_source_nachgereicht['Prozedurzeit'].replace(
        "00:00:00", np.nan
    )
    df_source_nachgereicht['OP-Code'] = df_source_nachgereicht['PrzCode']

    # 3. Concat all
    df_prozeduren_concat = pd.concat(
        [
            df_source,
            df_source_nachgereicht,
        ],
        ignore_index=True
    )

    df_prozeduren_3222 = df_prozeduren_concat[df_prozeduren_concat['OP-Code'] == '3-222']

    df_prozeduren_dedup = df_prozeduren_3222[[
        'Fallnummer', 'prozedur_datum', 'prozedur_zeit'
    ]].drop_duplicates().copy()
    df_fallnummern_dedup = df_prozeduren_concat.drop_duplicates(subset=['Fallnummer']).copy()
    print(f"{len(df_prozeduren_dedup)} von {len(df_prozeduren_3222)} 3-222-Prozeduren "
          f"sind eindeutig. Sie gehören zu {len(df_fallnummern_dedup)} Fallnummern.")

    return df_prozeduren_dedup