from dask import dataframe as dd
from get_source_dfs import get_labor_ddf


def add_laborwerte_to_prozeduren(df_prozeduren_final):
    ddf_labor = get_labor_ddf()
    ddf_prozeduren = dd.from_pandas(df_prozeduren_final, npartitions=1)

    # 1. Merge Labordaten auf die Prozeduren.
    #   Dabei fliegen alle Labordaten raus, deren Fallnummer nicht in der Prozedurentabelle vorkommt.
    #   Laborwerte werden immer dann zugewiesen, wenn ihr 'abnahmezeitpunkt_effektiv' innerhalb des
    #   Fensters liegt (also nach 'prozedur_fenster_start', aber vor 'prozedur_datetime'.
    #   Pro Prozedur wird es dadurch für jeden Laborwert eine Zeile geben.
    ddf_labor['Fallnummer'] = ddf_labor['Fallnummer'].astype('int64')
    ddf_prozeduren['Fallnummer'] = ddf_prozeduren['Fallnummer'].astype('int64')
    ddf_merged = dd.merge(
        ddf_prozeduren,
        ddf_labor,
        on=['Fallnummer'],
        how='left',
    )
    query_str = 'prozedur_fenster_start <= abnahmezeitpunkt_effektiv <= prozedur_datetime'
    ddf_labor_filtered = ddf_merged.query(query_str)

    return ddf_labor_filtered