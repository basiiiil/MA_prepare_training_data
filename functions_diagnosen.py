import datetime
import pandas as pd
import numpy as np
from dask import dataframe as dd

from config import CHARLSON_GROUPS


def get_diagnosen_df():
    print(datetime.datetime.now().strftime("%H:%M:%S") + " - Lese Diagnosen...")
    df_diagnosen = pd.read_csv(
        'fromDIZ/Diagnosen/25.07.2025_Diagnosen_sf.csv',
        usecols=[
            'Fall',
            'Diagnoseschlüssel',
            'DiagDatum',
            'Zeit'
        ],
        dtype={
            'Fall': np.int_,
            'Diagnoseschlüssel': np.str_,
            'DiagDatum': np.str_,
            'Zeit': np.str_
        }
    )
    # CSV zum Charlson Comorbidity Index laden
    df_charlson_mapping = pd.read_csv('icd_to_charlson_mapping.csv')
    df_charlson_mapping['icd_prefix'] = df_charlson_mapping['icd-10-code'].str.replace('.x', '', regex=False)
    df_charlson_mapping['prefix_length'] = df_charlson_mapping['icd_prefix'].str.len()
    # Mapping-Tabelle sortieren (absteigend nach Länge)
    # Dies ist SEHR WICHTIG, damit 'C91.10' (Länge 6) VOR 'C64' (Länge 3) gematcht wird.
    df_charlson_mapping = df_charlson_mapping.sort_values(by='prefix_length', ascending=False)

    # Den Diagnosen die jeweilige Komorbiditätsgruppe zuweisen
    df_diagnosen['charlson_group'] = pd.NA
    #    - Über die sortierte Mapping-Tabelle iterieren
    for index, row in df_charlson_mapping.iterrows():
        prefix = row['icd_prefix']
        comorbidity = row['charlson_group']

        # Eine Maske (Bedingung) erstellen für alle Diagnose-Codes, die:
        # 1. mit dem aktuellen Präfix beginnen
        # 2. NOCH KEINE Komorbidität zugewiesen bekommen haben (pd.NA)
        mask = (df_diagnosen['Diagnoseschlüssel'].str.startswith(prefix)) & (
            df_diagnosen['charlson_group'].isna())

        # Die Komorbidität für alle zutreffenden Zeilen eintragen
        df_diagnosen.loc[mask, 'charlson_group'] = comorbidity

    df_diagnosen['Fallnummer'] = df_diagnosen['Fall'].astype(int)
    df_diagnosen['diagnose_datetime'] = df_diagnosen['DiagDatum'] + "_" + df_diagnosen['Zeit']
    df_diagnosen['diagnose_datetime'] = pd.to_datetime(
        df_diagnosen['diagnose_datetime'], format='%m/%d/%Y_%I:%M:%S %p'
    )

    # Splitte Diagnoseschlüssel am Punkt
    # df_icd = df_diagnosen['Diagnoseschlüssel'].str.split('.', expand=True)
    # df_diagnosen['icd-dreistellig'] = df_icd[0]

    df_diagnosen_filtered = df_diagnosen[
        [
            'Fallnummer',
            'Diagnoseschlüssel',
            'charlson_group',
            'diagnose_datetime',
        ]
    ]
    df_diagnosen_dedup = df_diagnosen_filtered.drop_duplicates().copy()
    num_cases = df_diagnosen_dedup['Fallnummer'].nunique()
    print(f"{len(df_diagnosen_dedup)} Diagnosen zu {num_cases} Fallnummern gefunden.\n")

    return df_diagnosen_dedup

def get_prozedur_charlson_pivot(df_prozeduren):
    df_diagnosen = get_diagnosen_df()

    # Diagnosen auf Prozeduren mergen
    print(datetime.datetime.now().strftime("%H:%M:%S") + " - Merge Prozeduren und Diagnosen...")
    df_merged = pd.merge(
        df_prozeduren,
        df_diagnosen,
        on=['Fallnummer'],
        how='inner'
    ).reset_index(drop=True)
    print(f"Den {len(df_prozeduren)} Prozeduren wurden insgesamt {len(df_merged)} Diagnosen zugeordnet.")
    # df_merged = df_merged.sort_values(by=['Diagnoseschlüssel'], na_position='last')
    # df_merged_dedup = df_merged.drop_duplicates(subset=['Fallnummer', 'prozedur_datetime']).copy()
    # print(f"len(df_merged_dedup): {len(df_merged_dedup)}")

    query_str = "diagnose_datetime <= prozedur_datetime"
    df_merged_filtered = df_merged.query(query_str)

    # df_merged_filtered_dedup = df_merged_filtered.drop_duplicates(subset=['Fallnummer', 'prozedur_datetime']).copy()
    # print(f"len(df_merged_filtered_dedup): {len(df_merged_filtered_dedup)}")

    # Charlsongruppen zu Spalten konvertieren
    df_dummies = pd.get_dummies(df_merged_filtered['charlson_group'], prefix='charlson_group')
    df_result = pd.concat([df_merged_filtered[['Fallnummer', 'prozedur_datetime']], df_dummies], axis=1)

    col_names_pivot = CHARLSON_GROUPS.copy()
    col_names_pivot.extend(['Fallnummer', 'prozedur_datetime'])

    df_final = df_result.groupby(['Fallnummer', 'prozedur_datetime']).max().reset_index()
    df_final_small = df_final[col_names_pivot]
    # print(f"len(df_final): {len(df_final)}")

    return df_final_small