import datetime as dt
import pandas as pd
import numpy as np
from dask import dataframe as dd

"""
WAS DIESES SCRIPT TUN SOLL:
1. Laborwertzeilen filtern: Nur Zeilen mit folgenden Kriterien zulassen:
    1. Parameterbezeichnung ist nicht null
    2. Der Ergebniswert E ist eine Dezimalzahl (kein Text und keine Zeichen wie '<' oder '>')
    3. Die Referenzwerte sind Dezimalzahlen oder NaN
    4. Die Parameter-ID ist in der Definitionstabelle enthalten
    5. Die Referenzwerte sind gemäß der Definitionstabelle vorhanden
        Erklärung: Für jeden Parameter ist definiert, ob beide Referenzwerte, nur der obere oder untere,
        oder gar kein Referenzwert vorhanden sein muss.
        Wenn eine Zeile davon abweicht, ist sie nicht zugelassen.
        
2. Die Spalten 'Ergebniswert', 'Referenzwert unten' und 'Referenzwert oben' in numerische Werte umwandeln.
    
3. Die Ergebniswerte normalisieren:
   Die Parameter werden in vier Kategorien eingeteilt und dann die Werte entsprechend berechnet:
    0: ["beide"] Parameter mit oberem UND unterem Referenzwert
        (Ergebniswert E - unterer Referenzwert uR) / (oberer Referenzwert oR - uR)
    1: ["nur unten"] Parameter mit unterem, aber ohne oberem Referenzwert
        E / uR
    2: ["nur oben"] Parameter mit oberem, aber ohne unterem Referenzwert 
        E / oR
    3: ["ohne"] Parameter ohne Referenzwerte -> Absolutwert wird genutzt
"""

""" --- KONFIGURATION --- """
USE_MINI = False
USE_ORIGINAL_AND_CLEANUP = False
ANALYZE = False
FOLDER = 'Analyse Laborwerttabelle/'
FILENAME_FILTERED = 'Analyse Laborwerttabelle/2025-10-10_12-11_Laboruntersuchungen_only_valid.csv'

dtype_lib = {
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

def write_to_file(ddf, fn_out):
    filename = FOLDER + dt.datetime.now().strftime("%Y-%m-%d_%H-%M") + "_" + fn_out
    ddf.to_csv(
        filename,
        single_file=True,
        index=False,
    )

def add_numeric_columns(ddf, columns_labels_original):
    for column in columns_labels_original:
        column_label_new = column + '_num'
        ddf[column_label_new] = dd.to_numeric(ddf[column].str.replace(',', '.'), errors='coerce')
    return ddf

def get_dataframe_from_file(filename, variant):
    print(
        dt.datetime.now().strftime("%H:%M:%S")
        + f" - Lese '{filename}' (Variante: {variant})..."
    )
    if variant == 'mini':
        return dd.read_csv(
            filename,
            dtype=dtype_lib,
        )
    elif variant == 'original':
        df_col_names = dd.read_csv(
            'fromDIZ/20250929_LAE_Risiko_labor_Spaltennamen_AS.csv',
            delimiter=';',
            header=None,
            encoding='iso-8859-1'
        )
        col_names = df_col_names.head(1).squeeze().tolist()
        df_labor_original = dd.read_csv(
            filename,
            dtype=dtype_lib,
            usecols=dtype_lib.keys(),
            delimiter=';',
            decimal=',',
            header=None,
            names=col_names,
            encoding='utf_8',
            blocksize="64MB"
        )
        return df_labor_original
        # print("Entferne Zeilen mit Text und Punkten im Ergebniswert und Zeilen ohne Parameterbezeichnung...")
        # Entferne Zeilen, die nicht positive oder negative Dezimalzahlen sind. NA-Werte werden ebenfalls entfernt.
        # ergebnis_num = dd.to_numeric(df_labor_original['Ergebniswert'].str.replace(',', '.'), errors='coerce')
        # df_labor_clean = df_labor_original[
        #     ergebnis_num.notnull() & df_labor_original['Parameterbezeichnung'].notnull()
        # ]
        # len_original, len_clean = dd.compute(len(df_labor_original), len(df_labor_clean))
        # num_removed = len_original - len_clean
        # print(f"{num_removed} von {len_original} Zeilen mit fehlerhaftem Ergebniswert oder ohne Parameterbezeichnung entfernt.")

        # print(f"{num_false_referenzwerte} Zeilen mit fehlerhaften Referenzwerten entfernt.")
        # df_stats = pd.DataFrame({
        #     'Beschreibung': [
        #         'Zeilen original',
        #         'Zeilen mit nicht nutzbaren Ergebniswerten',
        #         'Zeilen mit nicht nutzbaren Referenzwerten'
        #     ],
        #     'Anzahl': [
        #         len(df_labor_original),
        #         num_false_ergebniswerte,
        #         num_false_referenzwerte,
        #     ]
        # })
        # df_stats.to_csv(FOLDER + now_string + '_stats.csv', index=False)
        # df_labor_clean.to_csv(
        #     FOLDER + dt.datetime.now().strftime("%H:%M:%S") + '_Laboruntersuchungen_only_valid.csv',
        #     single_file=True,
        #     index=False
        # )
        # return df_labor_clean
    elif variant == 'clean':
        print(f"{dt.datetime.now().strftime("%H:%M:%S")} - Lese CSV-Datei '{FILENAME_FILTERED}'...")
        return dd.read_csv(
            FILENAME_FILTERED,
            dtype=dtype_lib,
            decimal=',',
            blocksize="64MB"
        )
    else:
        raise ValueError(f"Variante '{variant}' nicht bekannt.")

def get_param_definition():
    return pd.read_csv(
        FOLDER + '2025-10-11_parameter_limits_definition.csv',
        usecols=[
            'Parameter-ID primär',
            'LOINC-Code',
            'Kategorie'
        ],
        dtype={
            'Kategorie': 'Int64'
        }
    )

def analyze_params(ddf):
    print("Analysiere Parameter und Einheiten...")
    unique_units = ddf[[
            'Parameterbezeichnung',
            'Ergebniseinheit',
            'Ergebniseinheit (UCUM)',
            'Parameter-ID primär',
            'LOINC-Code',
            'Referenzwert unten',
            'Referenzwert oben',
        ]].groupby(
        by=[
            'Parameterbezeichnung',
            'Ergebniseinheit',
            'Ergebniseinheit (UCUM)',
            'Parameter-ID primär',
            'LOINC-Code',
            'Referenzwert unten',
            'Referenzwert oben',
        ],
        group_keys=False,
        dropna=False
    ).size().reset_index().compute()
    unique_units.to_csv(
        FOLDER + dt.datetime.now().strftime("%H:%M:%S") + '_parameter_units_unique.csv',
        index=False,
    )

    df_units_per_parameter = ddf.groupby(
        'Parameterbezeichnung'
    )['Ergebniseinheit (UCUM)'].nunique().reset_index()
    df_units_per_parameter.compute()
    df_units_per_parameter.to_csv(
        FOLDER + dt.datetime.now().strftime("%H:%M:%S") + '_num_units_per_parameter.csv',
        index=False,
    )

def add_normalized_values(ddf, ):
    """
    Führt die Kategoriedefinitionen mit den Labordaten zusammen und berechnet
    die normalisierten Ergebniswerte basierend auf der jeweiligen Kategorie.
    """
    print(dt.datetime.now().strftime("%H:%M:%S") + " - Lade Parameter-Definitionen...")
    # Lade die Definitions-CSV mit Pandas (da sie klein ist)
    try:
        df_param_definition = get_param_definition()
    except FileNotFoundError:
        print(f"FEHLER: Die Definitionsdatei wurde nicht gefunden.")
        return ddf  # Gib den DataFrame unverändert zurück

    print(dt.datetime.now().strftime("%H:%M:%S") + " - Führe Labordaten und Definitionen zusammen...")
    # Führe die Definitionen mit dem Dask-DataFrame zusammen
    ddf = ddf.merge(df_param_definition, on=['Parameter-ID primär', 'LOINC-Code'], how='inner')

    print(dt.datetime.now().strftime("%H:%M:%S") + " - Berechne normalisierte Werte...")

    # --- Berechne die normalisierten Werte basierend auf den Kategorien ---
    cond_beide = (ddf['Kategorie'] == 0)
    cond_nur_unten = (ddf['Kategorie'] == 1)
    cond_nur_oben = (ddf['Kategorie'] == 2)
    cond_ohne = (ddf['Kategorie'] == 3)


    # 1. Initialisiere die neue Spalte mit NaN (Not a Number)
    ddf['Ergebniswert_normalisiert'] = np.nan

    # 2. Wende die Formeln konditional mit .mask() an
    # Formel für 'beide'
    range_width = ddf['Referenzwert oben_num'] - ddf['Referenzwert unten_num']
    ddf['Ergebniswert_normalisiert'] = ddf['Ergebniswert_normalisiert'].mask(
        cond_beide,
        (ddf['Ergebniswert_num'] - ddf['Referenzwert unten_num']) / range_width.replace(0, np.nan)
    )

    # Formel für 'nur_unten'
    ddf['Ergebniswert_normalisiert'] = ddf['Ergebniswert_normalisiert'].mask(
        cond_nur_unten,
        ddf['Ergebniswert_num'] / ddf['Referenzwert unten_num'].replace(0, np.nan)
    )

    # Formel für 'nur_oben'
    ddf['Ergebniswert_normalisiert'] = ddf['Ergebniswert_normalisiert'].mask(
        cond_nur_oben,
        ddf['Ergebniswert_num'] / ddf['Referenzwert oben_num'].replace(0, np.nan)
    )

    # Formel für 'ohne'
    ddf['Ergebniswert_normalisiert'] = ddf['Ergebniswert_normalisiert'].mask(
        cond_ohne,
        ddf['Ergebniswert_num']
    )

    return ddf

def merge_definitions(ddf):
    print(dt.datetime.now().strftime("%H:%M:%S") + " - Lade Parameter-Definitionen...")
    # Lade die Definitions-CSV mit Pandas (da sie klein ist)
    try:
        df_param_definition = get_param_definition()
    except FileNotFoundError:
        print(f"FEHLER: Die Definitionsdatei wurde nicht gefunden.")
        return ddf  # Gib den DataFrame unverändert zurück

    print(dt.datetime.now().strftime("%H:%M:%S") + " - Führe Labordaten und Definitionen zusammen...")
    # Führe die Definitionen mit dem Dask-DataFrame zusammen
    return ddf.merge(df_param_definition, on=['Parameter-ID primär', 'LOINC-Code'], how='inner')

def check_category_fit(ddf):
    """
    0: ["beide"]
    1: ["nur unten"]
    2: ["nur oben"]
    3: ["ohne"]
    """
    cond_beide = (
            (ddf['Kategorie'] == 0)
            & (ddf['Referenzwert unten'].notnull())
            & (ddf['Referenzwert oben'].notnull())
    )
    cond_unten = (
            (ddf['Kategorie'] == 1)
            & (ddf['Referenzwert unten'].notnull())
            & (~ddf['Referenzwert oben'].notnull())
    )
    cond_oben = (
            (ddf['Kategorie'] == 2)
            & (~ddf['Referenzwert oben'].notnull())
            & (ddf['Referenzwert oben'].notnull())
    )
    cond_ohne = (
            (ddf['Kategorie'] == 3)
            & (~ddf['Referenzwert unten'].notnull())
            & (~ddf['Referenzwert oben'].notnull())
    )
    return cond_beide | cond_unten | cond_oben | cond_ohne

def filter_for_relevant_rows(ddf):
    print(
        dt.datetime.now().strftime("%H:%M:%S")
        + " - Entferne Zeilen mit Text und Punkten im Ergebniswert und Zeilen ohne Parameterbezeichnung..."
    )
    # Erstelle numerische Spalten
    ddf['Ergebniswert_num'] = dd.to_numeric(ddf['Ergebniswert'].str.replace(',', '.'), errors='coerce')
    ddf['Referenzwert unten_num'] = dd.to_numeric(
        ddf['Referenzwert unten'].mask(ddf['Referenzwert unten'] == '0').str.replace(',', '.'),
        errors='coerce'
    )
    ddf['Referenzwert oben_num'] = dd.to_numeric(
        ddf['Referenzwert oben'].mask(ddf['Referenzwert oben'] == '0').str.replace(',', '.'),
        errors='coerce'
    )

    # Setze Kategoriespalte auf NaN, wenn Referenzwerte und Kategorie nicht übereinstimmen
    cond_beide = (
            (ddf['Kategorie'] == 0)
            & (ddf['Referenzwert unten'].notnull())
            & (ddf['Referenzwert oben'].notnull())
    )
    cond_unten = (
            (ddf['Kategorie'] == 1)
            & (ddf['Referenzwert unten'].notnull())
            & (~ddf['Referenzwert oben'].notnull())
    )
    cond_oben = (
            (ddf['Kategorie'] == 2)
            & (~ddf['Referenzwert oben'].notnull())
            & (ddf['Referenzwert oben'].notnull())
    )
    cond_ohne = (
            (ddf['Kategorie'] == 3)
            & (~ddf['Referenzwert unten'].notnull())
            & (~ddf['Referenzwert oben'].notnull())
    )

    ddf['Kategorie'] = ddf['Kategorie'].where(
        cond_beide | cond_unten | cond_oben | cond_ohne,
        np.nan
    )
    ddf_clean = ddf[
        ddf['Ergebniswert_num'].notnull()
        & ddf['Parameterbezeichnung'].notnull()
        & ddf['Kategorie'].notnull()
    ]
    len_original, len_clean = dd.compute(len(ddf), len(ddf_clean))
    num_removed = len_original - len_clean
    print(
        f"{num_removed} von {len_original} Zeilen mit Ergebniswert,"
        f"ohne Parameterbezeichnung oder mit falschen Referenzwerten entfernt."
    )

    return ddf_clean


def main():
    ddf_labor = get_dataframe_from_file(
        # filename=FOLDER + '2025-10-10_12-11_Laboruntersuchungen_only_valid.csv',
        filename='fromDIZ/20250929_LAE_Risiko_laboruntersuchungen_AS.csv',
        variant="original"
    )

    ddf_labor = merge_definitions(ddf_labor)
    print(ddf_labor.head())

    ddf_labor = filter_for_relevant_rows(ddf_labor)
    print(ddf_labor.head())

    """ ---------------------------------------------------- HIER WEITER ---------------------------------------------------- """

    # Füge Kategorien hinzu
    # ddf_labor = ddf_labor.merge(
    #     ddf_labor,
    # )
    #
    # # Schritt 2: Wandle 'Ergebniswert', 'Referenzwert unten' und 'Referenzwert oben' in Zahlenspalten um
    # print(dt.datetime.now().strftime("%H:%M:%S") + " - Füge Float-Spalten hinzu...")
    # ddf_labor = add_numeric_columns(
    #     ddf_labor, ['Ergebniswert', 'Referenzwert unten', 'Referenzwert oben']
    # )
    #
    # ddf_labor = add_normalized_values(ddf_labor)
    # write_to_file(ddf_labor, 'laborwerte_normalized.csv')



if __name__ == '__main__':
    main()