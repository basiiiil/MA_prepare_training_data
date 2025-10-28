import pandas as pd
import numpy as np

from functions_befunde import get_befunde_df
from functions_stammdaten import get_stammdaten_inpatients_df
from merge_data import merge_befunde_and_prozeduren


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

def get_prozeduren_for_labelling():
    # 1. Daten importieren
    df_stammdaten_inpatients = get_stammdaten_inpatients_df()
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

    print(f"Von {len(df_prozeduren)} Prozeduren sind "
          f"{len(df_prozeduren_inpatients)} von stationären Fällen.")

    # 3. Datums- und Zeitspalten in datetime Objekte umwandeln
    df_prozeduren_inpatients['prozedur_datetime'] = pd.to_datetime(
        df_prozeduren_inpatients['prozedur_datum'] + "_" + df_prozeduren_inpatients['prozedur_zeit'],
        format='%Y-%m-%d_%H:%M:%S',
    )
    # Alter für alle fehlenden Fälle berechnen, als Diff zwischen GebDatum und prozedur_datetime
    alter_aus_gebdatum = df_prozeduren_inpatients['alter_bei_prozedur'].fillna(
        (df_prozeduren_inpatients['prozedur_datetime'] - df_prozeduren_inpatients['geburtsdatum']).dt.days
    )
    alter_aus_gebdatum_float = alter_aus_gebdatum / 365.25
    df_prozeduren_inpatients['alter_bei_prozedur'] = df_prozeduren_inpatients['alter_bei_prozedur'].fillna(
        alter_aus_gebdatum_float.round().astype(int)
    )
    df_prozeduren_inpatients['altersdekade_bei_prozedur'] = np.ceil(df_stammdaten_inpatients['alter_bei_prozedur'] / 10)
    # 3.3 Entferne Prozeduren mit Alter < 18 Jahren
    df_prozeduren_no_minors = df_prozeduren_inpatients.query("alter_bei_prozedur >= 18").copy()
    print(f"Davon sind {len(df_prozeduren_no_minors)} mit alter_bei_prozedur >= 18.\n")

    return df_prozeduren_no_minors

