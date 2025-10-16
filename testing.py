import pandas as pd
import numpy as np

df = pd.read_csv(
    # 'Outputs/2025-10-15_Befunde_15to22_with_proz.csv',
    'Merged source files/2025-10-08_Prozeduren_3-222_with_behandlungsart.csv',
    dtype={
        'start_date': 'string',
        'start_time': 'string',
        'Behandlungsart': 'string',
    },
)

print(df.shape)
df_grouped = df.groupby('Fallnummer')[['Behandlungsart']].nunique()
print(df_grouped.value_counts())

df_fallnummern_unique = df.drop_duplicates(subset=['Fallnummer']).copy()
print(df_fallnummern_unique.shape)
df_fallnr_art_unique = df.drop_duplicates(subset=['Fallnummer', 'Behandlungsart']).copy()
print(df_fallnr_art_unique.shape)