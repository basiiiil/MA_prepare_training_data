from main import get_labelled_prozeduren
from merge_data import add_laborwerte_to_prozeduren
import datetime

print(datetime.datetime.now().strftime("%H:%M:%S") + " - Let's go!")
df_prozeduren_final = get_labelled_prozeduren()

# Laborwerte zu Prozeduren hinzufügen
ddf_analyse = add_laborwerte_to_prozeduren(df_prozeduren_final)
ddf_analyse['geburtsjahr'] = ddf_analyse['geburtsdatum'].dt.year

ddf_analyse_pd = ddf_analyse[[
    'Fallnummer',
    'parameterbezeichnung_effektiv',
    'parameterid_effektiv',
    'geburtsdatum',
    'alter_bei_prozedur',
    'altersdekade_bei_prozedur',
    'geschlecht'
]].copy().compute()

ddf_grouped = ddf_analyse_pd.groupby(
    by=[
        'parameterbezeichnung_effektiv',
        'parameterid_effektiv',
        'altersdekade_bei_prozedur',
        'geschlecht'
    ],
    as_index=False,
).size()
# ddf_grouped.to_csv(
#     'Analyse Laborwerttabelle/2025-10-17_analyse_dekade_geschlecht_2.csv',
#     index=False,
# )