import numpy as np
import pandas as pd
import re

from util_functions import concat_csv_files

'''
Zuletzt bearbeitet am 15.10.
Dieses Script reduziert die Liste der Befunde, sodass pro Fall pro Tag nur maximal ein Befund vorliegt.

1. Befunde werden identifiziert und Duplikate/mehrere Versionen gefunden.
    Wenn eine identifizierende Zeile im Kopf des Befundtextes (CONTENT) enthalten ist,
    wird die darin enthaltende Befundnummer genutzt, um Duplikate/Versionen zu identifizieren.
    Wenn nicht, wird die DOKNR und der gesamte Kopf des Befundtextes verglichen.
2. Bei mehreren Versionen wird nur die neueste Version erhalten.
3. Befunde, zu deren Fall ein weiterer Befund am selben Tag vorliegt, werden rausgeworfen.
'''

USEFUL_COLS = [
    'Fallnummer',
    'prozedur_datum',
    'befund_nummer',
    'befund_datum',
    'CONTENT_head',
    'CONTENT',
    'ZBEFALL04D',
    'DODAT',
    'ERDAT'
]

def get_doc_head_info(content_text):
    if pd.isna(content_text):
        return False
    lines = str(content_text).splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    pattern = r"\d{2}\.\d{2}\.\d{4}\s[-]+\s[A-Z0-9\-]{3,6}\s[-]+\s+[A-Za-z]{5,12}\s+[-]+\s+\d{1,9}\/\d{2}\s+[-]+"
    doc_head = "".join(non_empty_lines[:4])
    befundnummer = np.nan
    befunddatum = np.nan
    for line in non_empty_lines:
        if re.search(pattern, line):
            line_clean = re.sub(r"\s+[-]+\s*", " ", line)
            line_parts = line_clean.strip().split(' ')
            befundnummer = line_parts[-1]
            befunddatum = line_parts[0]
            doc_head = np.nan
            break

    return doc_head, befundnummer, befunddatum

def get_befunde_from_files():
    df_befunde = concat_csv_files(
        folder_path='fromDIZ/Befunde',
        csv_dtype={
            'Fallnummer': np.int_,
            'DOKNR': np.int_,
            'ZBEFALL04D': np.str_,
            'DODAT': np.str_,
            'ERDAT': np.str_,
        }
    )
    df_befunde['Fallnummer'] = df_befunde['FALNR']
    df_befunde['CONTENT_length'] = df_befunde['CONTENT'].apply(lambda x: len(x))
    df_befunde['datum_zbefall'] = pd.to_datetime(
        df_befunde['ZBEFALL04D'], format='%d.%m.%Y %H:%M:%S', errors='coerce'
    )
    df_befunde['jahr_zbefall'] = df_befunde['datum_zbefall'].dt.year
    df_befunde['datum_dodat'] = pd.to_datetime(
        df_befunde['DODAT'], format='%d.%m.%Y %H:%M:%S', errors='coerce'
    )
    df_befunde['datum_erdat'] = pd.to_datetime(
        df_befunde['ERDAT'], format='%d.%m.%Y %H:%M:%S', errors='coerce'
    )
    df_befunde[['CONTENT_head', 'befund_nummer', 'befund_datum_head']] = df_befunde.apply(
        lambda row: get_doc_head_info(row['CONTENT']), axis=1, result_type='expand'
    )
    df_befunde['befund_datum'] = pd.to_datetime(
        df_befunde['befund_datum_head'], format='%d.%m.%Y', errors='coerce'
    )
    df_befunde['befund_datum'] = df_befunde['befund_datum'].fillna(df_befunde['datum_erdat'])
    df_befunde['prozedur_datum'] = df_befunde.apply(
        lambda row: row['datum_zbefall'].strftime('%Y-%m-%d')
        if row['jahr_zbefall'] != 1900
        else row['datum_dodat'].strftime('%Y-%m-%d'),
        axis=1
    )

    return df_befunde

def dedup_befunde(df_befunde):
    # 1. Befunde mit identischer Fallnummer, Befundnummer und Prozedurdatum entfernen
    # 1a. Befunde nach Befunddatum sortieren, um immer den neuesten behalten zu können
    df_sorted = df_befunde.sort_values(by=['befund_datum'], ignore_index=True, ascending=False)

    # 1b. Duplikate entfernen und neuesten Befund (laut 'befund_datum') behalten.
    # Dabei kommt immer dann 'befund_nummer' zum Tragen, wenn der Befund eine identifizierende Zeile enthält.
    # Besp.: '13.08.2015 ------ RAD -----  Arztbrief  -----  79572/15  ----------------------'
    # Wenn so eine Zeile nicht existiert, müssen die ersten 4 Zeilen ('CONTENT_head') identisch sein,
    # damit ein Duplikat als solches gilt.
    df_befunde_dedup = df_sorted.drop_duplicates(
        subset=['Fallnummer', 'prozedur_datum', 'befund_nummer', 'CONTENT_head'],
        keep='first',
    ).copy()

    return df_befunde_dedup

def get_unique_befunde_per_case_day(df_befunde_dedup):
    # Befunde entfernen, bei denen mehrere Befunde zum gleichen Fall und zum gleichen Tag,
    # aber mit unterschiedlicher Befundnummer und unterschiedlichem Befundtext existieren
    df_grouped = df_befunde_dedup.groupby(
        by=['Fallnummer', 'prozedur_datum'],
        as_index=False,
        dropna=False,
    )[['befund_nummer']].size()
    df_merged = pd.merge(
        df_befunde_dedup, df_grouped, on=['Fallnummer', 'prozedur_datum'], how='left'
    )
    df_befunde_unique_caseday = df_merged[df_merged['size'] == 1]

    return df_befunde_unique_caseday

def get_befunde_df():
    df_source = get_befunde_from_files()

    df_befunde_dedup = dedup_befunde(df_source)

    df_befunde_unique_case_day = get_unique_befunde_per_case_day(df_befunde_dedup)
    print(f"\nEs existieren {len(df_befunde_unique_case_day)} Befunde, an deren Prozedurdatum "
          f"zu diesem Fall kein anderer Befund vorliegt. Somit ist eine eindeutige Zuordung "
          f"zu einer Prozedur möglich.")

    return df_befunde_unique_case_day
