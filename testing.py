from dask import dataframe as dd
import numpy as np

df_bga = dd.read_csv(
    'fromDIZ/Laborwerte/20251024_LAE_Risiko_laboruntersuchungen_Blutgase_AS.csv',
    sep=';',
    usecols=[
        'Fallnummer',
        'Probeneingangsdatum',
        'Ergebnisdatum',
        'Parameter-ID primär'
    ],
    dtype={
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
df_bga['zeit_diff_seconds_abs'] = df_bga['zeit_diff_seconds'].abs()
df_bga['mehr_als_12h'] = df_bga['zeit_diff_seconds_abs'] > 43200 # 86400
df_bga['mehr_als_24h'] = df_bga['zeit_diff_seconds_abs'] > 86400
df_bga['mehr_als_5d'] = df_bga['zeit_diff_seconds_abs'] > 432000
df_bga['mehr_als_130d'] = df_bga['zeit_diff_seconds_abs'] > 11232000
df_bga['mehr_als_1h'] = df_bga['zeit_diff_seconds_abs'] > 3600
df_bga['mehr_als_15min'] = df_bga['zeit_diff_seconds_abs'] > 900
df_bga['max_15min'] = df_bga['zeit_diff_seconds_abs'] <= 900
df_bga['max_20min'] = df_bga['zeit_diff_seconds_abs'] <= 1200
df_bga['max_30min'] = df_bga['zeit_diff_seconds_abs'] <= 1800
df_bga['max_60min'] = df_bga['zeit_diff_seconds_abs'] <= 3600
df_bga['max_120min'] = df_bga['zeit_diff_seconds_abs'] <= 7200
df_bga['max_3h'] = df_bga['zeit_diff_seconds_abs'] <= 3 * 3600
df_bga['max_4h'] = df_bga['zeit_diff_seconds_abs'] <= 4 * 3600
df_bga['max_6h'] = df_bga['zeit_diff_seconds_abs'] <= 6 * 3600
df_bga['max_12h'] = df_bga['zeit_diff_seconds_abs'] <= 12 * 3600
df_bga['ergebnis_nach_eingang'] = df_bga['zeit_diff_seconds'] >= 0

df_eingang_vor_ergebnis = df_bga[df_bga['ergebnis_nach_eingang']]
df_ergebnis_vor_eingang = df_bga[~df_bga['ergebnis_nach_eingang']]

print("Start computing...")
# params_grouped = df_bga.groupby(
#     by='Parameter-ID primär',
# ).agg(
#     sum=('ergebnis_nach_eingang', 'sum'),
#     count=('ergebnis_nach_eingang', 'count'),
#     max=('zeit_diff_seconds_abs', 'max'),
#     mean=('zeit_diff_seconds_abs', 'mean'),
#     max_15min=('max_15min', 'sum'),
#     max_20min=('max_20min', 'sum'),
#     max_30min=('max_30min', 'sum'),
#     max_60min=('max_60min', 'sum'),
#     max_120min=('max_120min', 'sum'),
# ).compute()
#
# print(params_grouped.head())
# params_grouped.to_csv('params_grouped_5.csv')

# df_eingang_vor_ergebnis_grouped = df_eingang_vor_ergebnis.groupby(
#     by='Parameter-ID primär',
# ).agg(
#     count=('ergebnis_nach_eingang', 'count'),
#     max=('zeit_diff_seconds_abs', 'max'),
#     mean=('zeit_diff_seconds_abs', 'mean'),
#     max_15min=('max_15min', 'sum'),
#     max_20min=('max_20min', 'sum'),
#     max_30min=('max_30min', 'sum'),
#     max_60min=('max_60min', 'sum'),
#     max_120min=('max_120min', 'sum'),
# ).compute()
# df_eingang_vor_ergebnis_grouped.to_csv('df_eingang_vor_ergebnis_grouped_2.csv')

df_ergebnis_vor_eingang_grouped = df_ergebnis_vor_eingang.groupby(
    by='Parameter-ID primär',
).agg(
    count=('ergebnis_nach_eingang', 'count'),
    max=('zeit_diff_seconds_abs', 'max'),
    mean=('zeit_diff_seconds_abs', 'mean'),
    max_15min=('max_15min', 'sum'),
    max_20min=('max_20min', 'sum'),
    max_30min=('max_30min', 'sum'),
    max_60min=('max_60min', 'sum'),
    max_120min=('max_120min', 'sum'),
    max_3h=('max_3h', 'sum'),
    max_4h=('max_4h', 'sum'),
    max_6h=('max_6h', 'sum'),
    max_12h=('max_12h', 'sum'),
).compute()
df_ergebnis_vor_eingang_grouped.to_csv('df_ergebnis_vor_eingang_grouped_3.csv')

