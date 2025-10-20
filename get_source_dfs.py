import numpy as np
import pandas as pd

from functions_labor import filter_for_relevant_rows, normalize_ids_and_timestamps, filter_for_relevant_rows_pandas, \
    normalize_ids_and_timestamps_pandas
from functions_befunde import dedup_befunde, get_unique_befunde_per_case_day, get_befunde_from_files
from util_functions import get_complete_laborwerte_ddf, get_complete_laborwerte_df_pandas


def get_stammdaten_inpatients_df():
    """
    :return: pandas dataframe
    """
    df_source = pd.read_csv(
        'fromDIZ/Stammdaten/24.07.2025_Patientenliste_3-222_Stammdaten_sf.csv',
        dtype={
            'Fallnummer': np.int_,
            # 'Patientennummer': np.int_,
            'Alter': np.int_,
            # 'Aufnahmedatum': np.str_,
        },
        usecols=[
            'Fallnummer',
            # 'Patientennummer',
            'Geschlecht',
            'Alter',
            'Bewegung Behandlungsart',
            # 'Aufnahmedatum'
        ]
    )

    df_source_nachgereicht = pd.read_csv(
        'fromDIZ/Stammdaten/Stammdaten.csv',
        dtype={
            # 'Patient': np.int_,
            'Fall': np.int_,
            'GebDatum': np.str_
        },
        usecols=[
            'Fall',
            # 'Patient',
            'G',
            'GebDatum',
            'Fa'
        ]
    )

    df_source['behandlungsart'] = df_source['Bewegung Behandlungsart']
    df_source['geschlecht'] = df_source['Geschlecht']
    # df_source['geburtsdatum'] = np.nan
    df_source['alter_bei_prozedur'] = df_source['Alter']

    df_source_nachgereicht['Fallnummer'] = df_source_nachgereicht['Fall']
    # df_source_nachgereicht['Patientennummer'] = df_source_nachgereicht['Patient']
    df_source_nachgereicht['behandlungsart'] = df_source_nachgereicht['Fa']
    df_source_nachgereicht['geschlecht'] = df_source_nachgereicht['G']
    df_source_nachgereicht['geburtsdatum'] = pd.to_datetime(
        df_source_nachgereicht['GebDatum'], format='%Y-%m-%d'
    )
    # df_source_nachgereicht['alter_bei_prozedur'] = np.nan

    # 3. Merge all Fallnummern and drop duplicates
    df_stammdaten_all = pd.concat(
        [
            df_source[
                [
                    'Fallnummer',
                    'geschlecht',
                    # 'geburtsdatum',
                    'alter_bei_prozedur',
                    'behandlungsart'
                ]
            ],
            df_source_nachgereicht[
                [
                    'Fallnummer',
                    'geschlecht',
                    'geburtsdatum',
                    # 'alter_bei_prozedur',
                    'behandlungsart'
                ]
            ],
        ]
    )
    # df_stammdaten_all = df_source[
    #     [
    #         'Fallnummer',
    #         'geschlecht',
    #         # 'geburtsdatum',
    #         'alter_bei_prozedur',
    #         'behandlungsart'
    #     ]
    # ]
    df_stammdaten_all_inpatients = df_stammdaten_all[
        df_stammdaten_all['behandlungsart'] == 'Stationär'
    ].reset_index(drop=True)
    df_stammdaten_dedup = df_stammdaten_all_inpatients.drop_duplicates().copy()
    print(f"Es existieren Stammdaten zu {len(df_stammdaten_dedup)} stationären Fällen.")

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
            'Fall': np.int_,
            'OP-Code Beg.': np.str_,
            'BegZeit': np.str_
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
            'Fallnummer': np.int_,
            'Prozedurzeit': np.str_
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

def get_labor_ddf():
    ddf_labor = get_complete_laborwerte_ddf()
    ddf_labor_filtered = filter_for_relevant_rows(ddf_labor)
    ddf_labor_normalized = normalize_ids_and_timestamps(ddf_labor_filtered)

    return ddf_labor_normalized

def get_labor_df_pandas():
    df_labor = get_complete_laborwerte_df_pandas()
    df_labor_filtered = filter_for_relevant_rows_pandas(df_labor)
    ddf_labor_normalized = normalize_ids_and_timestamps_pandas(df_labor_filtered)

    return ddf_labor_normalized