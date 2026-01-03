import pandas as pd

from config import LABOR_PARAMS_EFFEKTIV_SMALL

df_final = pd.read_csv('2025-11-02_15:46_proz_mit_labor_und_diagnosen_final_WITH_GLU.csv')
df_final_complete = df_final.dropna(subset=LABOR_PARAMS_EFFEKTIV_SMALL)

print("\n--- Stats alle ---\n")

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



