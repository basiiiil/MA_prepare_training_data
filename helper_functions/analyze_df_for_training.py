import pandas as pd

from config import LABOR_PARAMS_EFFEKTIV_SMALL

df_final = pd.read_csv('../2025-11-02_15:46_proz_mit_labor_und_diagnosen_final_WITH_GLU.csv')
df_final_complete = df_final.dropna(subset=LABOR_PARAMS_EFFEKTIV_SMALL)

print("\n--- Stats alle ---\n")
print(f"Num columns df_final: {len(df_final.columns)}")
print(f"Num columns df_final_complete: {len(df_final_complete.columns)}")

print(f"Ereignisse gesamt: {len(df_final)}")
print(f"Patient:innen gesamt: {df_final["Patientennummer"].nunique()}")


print(df_final['predicted_label_int'].value_counts(dropna=False))
print(df_final['predicted_label_int'].value_counts(dropna=False, normalize=True))

print(df_final['geschlecht'].value_counts(dropna=False))
print(df_final['geschlecht'].value_counts(dropna=False, normalize=True))

print(f"Unique Patients: {df_final['Patientennummer'].nunique()}")

print("\n--- Stats complete ---\n")

print(f"Ereignisse complete: {len(df_final_complete)}")
print(f"Patient:innen complete: {df_final_complete['Patientennummer'].nunique()}")

print(df_final_complete['predicted_label_int'].value_counts(dropna=False))
print(df_final_complete['predicted_label_int'].value_counts(dropna=False, normalize=True))

print(df_final_complete['geschlecht'].value_counts(dropna=False))
print(df_final_complete['geschlecht'].value_counts(dropna=False, normalize=True))

print(f"Unique Patients: {df_final_complete['Patientennummer'].nunique()}")

print("\n---------------\n")

print(f"--- {len(df_final)} - {len(df_final_complete)} "
      f"= {len(df_final) - len(df_final_complete)} ---")

print("\n---------------\n")

for i in range(17):
    print(f"CCI {i+1} count: {df_final_complete[f'charlson_group_{i+1}'].sum()}")

df_final_complete['has_any_cci'] = (df_final_complete['charlson_group_1']
                           | df_final_complete['charlson_group_2']
                           | df_final_complete['charlson_group_3']
                           | df_final_complete['charlson_group_4']
                           | df_final_complete['charlson_group_5']
                           | df_final_complete['charlson_group_6']
                           | df_final_complete['charlson_group_7']
                           | df_final_complete['charlson_group_8']
                           | df_final_complete['charlson_group_9']
                           | df_final_complete['charlson_group_10']
                           | df_final_complete['charlson_group_11']
                           | df_final_complete['charlson_group_12']
                           | df_final_complete['charlson_group_13']
                           | df_final_complete['charlson_group_14']
                           | df_final_complete['charlson_group_15']
                           | df_final_complete['charlson_group_16']
                           | df_final_complete['charlson_group_17']
                           )

print(f"Num CTs without CCI: {df_final_complete['has_any_cci'].sum()}")

df_final_complete['num_missing'] = df_final_complete.isna().sum(axis=1).copy()
df_final_complete['num_present'] = (69 - df_final_complete['num_missing']).copy()
print(f"Nan: {df_final_complete['num_missing'].describe()}")
print(df_final_complete['num_present'].describe())
