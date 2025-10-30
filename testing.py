import pandas as pd
from dask import dataframe as dd
import matplotlib.pyplot as plt
import datetime

from functions_labor import get_labor_ddf, get_complete_laborwerte_ddf, filter_for_relevant_rows, \
    normalize_ids_and_timestamps, calculate_effective_time
from main import get_now_label

def get_blutgas_histogram():
    df_bga = dd.read_csv(
        'fromDIZ/Laborwerte/20251024_LAE_Risiko_laboruntersuchungen_Blutgase_AS.csv',
        sep=';',
        usecols=[
            'Auftragsnummer',
            'Fallnummer',
            'Probeneingangsdatum',
            'Ergebnisdatum',
            'Parameter-ID primär'
        ],
        dtype={
            'Auftragsnummer': float,
            'Fallnummer': int,
            'Probeneingangsdatum': str,
            'Ergebnisdatum': str,
            'Parameter-ID primär': str
        },
        decimal=',',
        encoding='utf_8',
        blocksize="64MB"
    )

    df_bga['dt_eingang'] = dd.to_datetime(df_bga['Probeneingangsdatum'], format='%Y-%m-%dT%H:%M:%S%z')
    df_bga['dt_eingang'] = df_bga['dt_eingang'].dt.tz_localize(None)
    df_bga['dt_ergebnis'] = dd.to_datetime(df_bga['Ergebnisdatum'], format='%Y-%m-%dT%H:%M:%S%z')
    df_bga['dt_ergebnis'] = df_bga['dt_ergebnis'].dt.tz_localize(None)

    df_bga['zeit_diff_td'] = df_bga['dt_ergebnis'] - df_bga['dt_eingang']
    df_bga['zeit_diff_seconds'] = df_bga['zeit_diff_td'].dt.total_seconds()
    df_bga['zeit_diff_mins'] = df_bga['zeit_diff_seconds'] / 60
    df_bga['zeit_diff_mins_max6h'] = df_bga['zeit_diff_mins'].apply(
        lambda x: x if x < 360 else 360, meta=('zeit_diff_mins', 'float64')
    )
    df_bga['zeit_diff_seconds_abs'] = df_bga['zeit_diff_seconds'].abs()
    df_bga['ergebnis_nach_eingang'] = df_bga['zeit_diff_seconds'] >= 0

    df_bga['fenster_5_min'] = (df_bga['zeit_diff_seconds'] // 300) * 5
    df_bga['fenster_1h'] = (df_bga['zeit_diff_seconds'] // 3600) * 60

    df_eingang_vor_ergebnis = df_bga[df_bga['ergebnis_nach_eingang']]
    df_ergebnis_vor_eingang = df_bga[~df_bga['ergebnis_nach_eingang']]

    print("Start computing...")
    df_bga_filtered = df_bga[df_bga['zeit_diff_seconds'].notnull()]
    # df_bga_filtered_2 = df_bga[
    #     (df_bga['zeit_diff_seconds'] <= (6*3600))
    #     & (df_bga['zeit_diff_seconds'] >= (-6 * 3600))
    # ]

    df_bga_filtered_small = df_bga_filtered[['Auftragsnummer', 'zeit_diff_mins_max6h']].compute()

    # group = df_bga_filtered.groupby('fenster_5_min').size()
    # print(group.head())

    # group.to_csv(
    #     get_now_label() + 'bga_5min_fenster.csv',
    #     single_file=True,
    # )

    df_bga_filtered_small['zeit_diff_mins_max6h'].hist(
        bins=100,  # Anzahl der Intervalle (Bins),
        edgecolor='black' # Optional: Ränder für bessere Sichtbarkeit
    )

    # 3. Diagramm beschriften und anzeigen
    plt.title('BGA: Minuten zwischen Probeneingang und Ergebnis')
    plt.xlabel('Minuten')
    plt.ylabel('Anzahl Einträge')
    plt.grid(axis='y', alpha=0.75) # Optional: Gitterlinien

    output_filename = get_now_label() + "_bga_histogramm.png"
    plt.savefig(output_filename)
    plt.close()
    print(f"\nGrafik wurde erfolgreich als '{output_filename}' gespeichert.")

def get_blutgas_zeitverteilung():
    df_bga = dd.read_csv(
        'fromDIZ/Laborwerte/20251024_LAE_Risiko_laboruntersuchungen_Blutgase_AS.csv',
        sep=';',
        usecols=[
            'Auftragsnummer',
            'Fallnummer',
            'Probeneingangsdatum',
            'Ergebnisdatum',
            'Parameter-ID primär'
        ],
        dtype={
            'Auftragsnummer': float,
            'Fallnummer': int,
            'Probeneingangsdatum': str,
            'Ergebnisdatum': str,
            'Parameter-ID primär': str
        },
        decimal=',',
        encoding='utf_8',
        blocksize="64MB"
    )
    df_bga['Probeneingangsdatum'] = df_bga['Probeneingangsdatum'].fillna(
        df_bga['Ergebnisdatum']
    )
    df_bga['dt_eingang'] = dd.to_datetime(df_bga['Probeneingangsdatum'], format='%Y-%m-%dT%H:%M:%S%z')
    df_bga['year'] = df_bga['dt_eingang'].dt.year

    grouped = df_bga.groupby(['Parameter-ID primär', 'year']).size()

    grouped.to_csv(
        get_now_label() + "_bga_grouped_years.csv",
        single_file=True,
    )

def get_unique_laborparameter():
    # ddf_labor = get_labor_ddf('complete')
    # df_map = pd.read_csv('labor_parameterid_bezeichnung_map.csv')

    # df_labor_params = ddf_labor[['parameterid_effektiv']].drop_duplicates().copy().compute()

    # df_labor_params['parameterbezeichnung_effektiv'] = df_labor_params.where(
    #     df_labor_params['parameterid_effektiv'].isin(df_map['parameterid_effektiv'])
    # )

    ddf_labor = get_complete_laborwerte_ddf()
    ddf = filter_for_relevant_rows(ddf_labor, 'complete')
    # ddf_final = normalize_ids_and_timestamps(ddf)

    # ddf['length_ped'] = ddf['Probeneingangsdatum'].str.len()
    # ddf['ped_replaced'] = ddf['Probeneingangsdatum'].str.replace(r'\d', 'x', regex=True)

    # 1. 'abnahmezeitpunkt_effektiv' für alle Werte setzen.
    # Für AccuCheck wird die Zeit aus der Parameterbezeichnung genutzt
    ddf['Probeneingangsdatum'] = dd.to_datetime(
        ddf['Probeneingangsdatum'],
        format='%Y-%m-%dT%H:%M:%S%z',
        utc=True,
    )
    ddf['Probeneingangsdatum'] = ddf['Probeneingangsdatum'].dt.tz_localize(None)
    ddf['probeneingang_tag'] = ddf['Probeneingangsdatum'].dt.floor('D')

    ddf['accuchek_strings'] = ddf['Parameter-ID primär'].where(
        ddf['Parameter-ID primär'].str.contains(r"O-GLU_Z\.\d{4}", regex=True, na=False)
    )
    ddf['accuchek_strings'] = ddf['accuchek_strings'].str.split('.', n=1, expand=True)[1]
    ddf['accuchek_hours'] = ddf['accuchek_strings'].str[:2].astype('f8')
    ddf['accuchek_mins'] = ddf['accuchek_strings'].str[2:4].astype('f8')
    ddf['ac_timedelta'] = dd.to_timedelta(
        ddf['accuchek_hours'], unit='h'
    ) + dd.to_timedelta(ddf['accuchek_mins'], unit='m')
    # ddf['ac_day'] = ddf['Probeneingangsdatum'].dt.floor('D')
    # ddf['abnahmezeitpunkt_effektiv'] = ddf['ac_day'] + ddf['ac_timedelta']
    ddf['abnahmezeitpunkt_effektiv'] = ddf['Probeneingangsdatum'].mask(
        ddf['Parameter-ID primär'].str.contains(r"O-GLU_Z\.\d{4}", regex=True, na=False),
        ddf['probeneingang_tag'] + ddf['ac_timedelta']
    )

    print(ddf.columns)

    print(datetime.datetime.now().strftime("%H:%M:%S") + " - Start computing...")
    df_pandas = ddf.compute()
    print(datetime.datetime.now().strftime("%H:%M:%S") + " - Computing fertig.\n\n")

    print(df_pandas.shape)

    # print(df_pandas['length_ped'].value_counts())
    # print(df_pandas['ped_replaced'].value_counts())
    print(datetime.datetime.now().strftime("%H:%M:%S") + " ----- All done -----")

def test_datetime_conversions():
    df = pd.DataFrame({
        'Probeneingangsdatum': [
            '2014-12-06T14:39:00+0100',
            '2014-11-19T16:47:00+01:00',
            '2014-10-14T13:37:00+0200',
            '2015-12-31T12:32:00+02:00',
            '2014-10-14T13:37:00+0200',
            '2015-12-31T12:32:00+0100',
        ],
        'Parameter-ID primär': [
            'O-GLU_Z.0045',
            'G-GLU_K',
            'GGT_S',
            'GLU_S',
            'O-GLU_Z',
            'O-GLU_Z.2315',
        ]
    })
    ddf = dd.from_pandas(df)

    ddf['Probeneingangsdatum'] = dd.to_datetime(
        ddf['Probeneingangsdatum'],
        format='%Y-%m-%dT%H:%M:%S%z',
        utc=True,
    )
    ddf['Probeneingangsdatum'] = ddf['Probeneingangsdatum'].dt.tz_localize(None)
    ddf['probeneingang_tag'] = ddf['Probeneingangsdatum'].dt.floor('D')

    ddf['accuchek_strings'] = ddf['Parameter-ID primär'].where(
        ddf['Parameter-ID primär'].str.contains(r"O-GLU_Z\.\d{4}", regex=True, na=False)
    )
    ddf['accuchek_strings'] = ddf['accuchek_strings'].str.split('.', n=1, expand=True)[1]
    ddf['accuchek_hours'] = ddf['accuchek_strings'].str[:2].astype('f8')
    ddf['accuchek_mins'] = ddf['accuchek_strings'].str[2:4].astype('f8')
    ddf['ac_timedelta'] = dd.to_timedelta(
        ddf['accuchek_hours'], unit='h'
    ) + dd.to_timedelta(ddf['accuchek_mins'], unit='m')
    ddf['abnahmezeitpunkt_effektiv'] = ddf['Probeneingangsdatum'].mask(
        ddf['Parameter-ID primär'].str.contains(r"O-GLU_Z\.\d{4}", regex=True, na=False),
        ddf['probeneingang_tag'] + ddf['ac_timedelta']
    )

    df_pandas = ddf.compute()
    print(df_pandas)

def main():
    ddf = get_labor_ddf('complete')
    print(ddf.columns)

    print(datetime.datetime.now().strftime("%H:%M:%S") + " - Start computing...")
    df_pandas = ddf.compute()
    print(datetime.datetime.now().strftime("%H:%M:%S") + " - Computing fertig.\n\n")

    print(df_pandas.shape)


if __name__ == '__main__':
    main()