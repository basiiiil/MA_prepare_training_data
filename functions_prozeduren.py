import pandas as pd
import numpy as np

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
