import pandas as pd

from config import LABOR_PARAMS_EFFEKTIV_SMALL

df_final = pd.read_csv('Outputs/2025-10-28_proz_mit_labor_und_diagnosen_final.csv')

print(df_final['predicted_label_int'].value_counts(dropna=False))
print(df_final['predicted_label_int'].value_counts(dropna=False, normalize=True))

print(df_final['geschlecht'].value_counts(dropna=False))
print(df_final['geschlecht'].value_counts(dropna=False, normalize=True))

print(f"NaNs in alter_bei_prozedur: {df_final['alter_bei_prozedur'].isna().sum()}")
print(f"NaNs in prozedur_datetime: {df_final['prozedur_datetime'].isna().sum()}")

print(f"--- {len(df_final)} - {len(df_final.dropna(subset=LABOR_PARAMS_EFFEKTIV_SMALL))} "
      f"= {len(df_final) - len(df_final.dropna(subset=LABOR_PARAMS_EFFEKTIV_SMALL))} ---")
