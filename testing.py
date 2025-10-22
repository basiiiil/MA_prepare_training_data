from operator import index

from dask import dataframe as dd
import pandas as pd
import numpy as np

from get_source_dfs import get_labor_ddf, get_stammdaten_inpatients_df
from main import get_labelled_prozeduren, get_latest_lab_values, get_time_window_table, get_now_label
from merge_data import add_laborwerte_to_prozeduren
import datetime

print(datetime.datetime.now().strftime("%H:%M:%S") + " - Let's go!")

ddf_labor = get_labor_ddf()
df_labor_cases = ddf_labor.drop_duplicates(subset=['Fallnummer']).copy().compute()

df_stammdaten = get_stammdaten_inpatients_df()
num_cases_stammdaten = df_stammdaten['Fallnummer'].nunique()

num_cases_labor = df_labor_cases['Fallnummer'].nunique()
df_cases_not_in_labor = df_stammdaten[~df_stammdaten['Fallnummer'].isin(df_labor_cases['Fallnummer'])].copy()
num_cases_not_in_labor = len(df_cases_not_in_labor)
df_cases_not_in_stammdaten = df_labor_cases[~df_labor_cases['Fallnummer'].isin(df_stammdaten['Fallnummer'])].copy()
num_cases_not_in_stammdaten = len(df_cases_not_in_stammdaten)


print(f"Cases in Stammdaten: {num_cases_stammdaten}")
print(f"Cases in Laborwerte: {num_cases_labor}")
print(f"Cases in Stammdaten, but not in Labor: {num_cases_not_in_labor}")
print(f"Cases in Labor, but not in Stammdaten: {num_cases_not_in_stammdaten}")

