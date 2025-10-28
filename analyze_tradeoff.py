import datetime
import pandas as pd
from dask import dataframe as dd

from functions_labor import get_labor_ddf, get_latest_lab_values
from main import get_prozeduren_for_training, get_now_label

# 1. Hole Prozeduren, mit Zeitfenster gesamt = 7 Tage * 24h = 168h
print(datetime.datetime.now().strftime("%H:%M:%S") + " - Let's go!")
# 1. Hole Prozeduren
df_prozeduren = get_prozeduren_for_training()

# 2. Merge Labor auf Prozeduren
df_prozeduren['prozedur_fenster_start'] = pd.to_datetime(
    df_prozeduren['prozedur_datetime'] - pd.Timedelta(hours=168),
)

ddf_labor = get_labor_ddf(variant='complete')

# 1. Merge Labordaten auf die Prozeduren.
#   Dabei fliegen alle Labordaten raus, deren Fallnummer nicht in der Prozedurentabelle vorkommt.
#   Laborwerte werden immer dann zugewiesen, wenn ihr 'abnahmezeitpunkt_effektiv' innerhalb des
#   Fensters liegt (also nach 'prozedur_fenster_start', aber vor 'prozedur_datetime'.
#   Pro Prozedur wird es dadurch für jeden Laborwert eine Zeile geben.
ddf_labor['Fallnummer'] = ddf_labor['Fallnummer'].astype('int32')
df_prozeduren['Fallnummer'] = df_prozeduren['Fallnummer'].astype('int32')
ddf_merged = dd.merge(
    df_prozeduren,
    ddf_labor,
    on=['Fallnummer'],
    how='inner',
).reset_index(drop=True)

# 2. Filtere auf Laborwerte, die innerhalb des Laborfensters liegen
query_str = 'prozedur_fenster_start <= abnahmezeitpunkt_effektiv <= prozedur_datetime'
ddf_labor_filtered = ddf_merged.query(query_str)

# 3. Filtere auf die jeweils letzten Werte je Parameter je Prozedur
ddf_labor_latest = get_latest_lab_values(ddf_labor_filtered)

# 4. Minimiere die Größe des Dataframes
ddf_labor_latest_small = ddf_labor_latest[[
    'Fallnummer',
    'prozedur_datetime',
    'parameterid_effektiv',
    'ergebniswert_num'
]]

# --- 2. Parameter-Häufigkeit (Prävalenz) ---

# Zählen, wie viele einzigartige Fälle jeden Parameter haben
print(datetime.datetime.now().strftime("%H:%M:%S") + " - Computing...")
param_prevalence = ddf_labor_latest_small.groupby('parameterid_effektiv').size().compute()

# Sortieren: Die häufigsten Parameter zuerst
sorted_params = param_prevalence.sort_values(ascending=False).index.tolist()
# --- 3. Parameter-Sets pro Fall erstellen (Effiziente Methode) ---

# Erstellt für jede fallprozedur_id ein Set all ihrer parameter
case_param_sets = ddf_labor_latest_small.groupby(
    by=['Fallnummer', 'prozedur_datetime']
)['parameterid_effektiv'].apply(set).compute()
# Gesamtzahl der einzigartigen Fälle
total_cases = len(case_param_sets)
print(f"Gesamtzahl einzigartiger Fälle: {total_cases}\n")

# --- 4. Trade-off-Analyse durchführen ---

analysis_results = []
core_set = set()

print("Starte Trade-off-Analyse...")
# Iterieren durch die sortierte Parameterliste
for k, param_id in enumerate(sorted_params, 1):

    # Den neuen Parameter zum Kernset hinzufügen
    core_set.add(param_id)

    # Zählen, wie viele Fälle das *gesamte* Kernset enthalten
    # set_A.issubset(set_B) prüft, ob alle Elemente von A auch in B sind
    num_complete_cases = case_param_sets.apply(lambda case_set: core_set.issubset(case_set)).sum()

    # Ergebnisse speichern
    percent_cases_kept = (num_complete_cases / total_cases) * 100
    percent_cases_excluded = 100.0 - percent_cases_kept

    analysis_results.append({
        'k_num_params': k,
        'last_param_added': param_id,
        'num_cases_kept': num_complete_cases,
        'percent_cases_kept': percent_cases_kept,
        'percent_cases_excluded': percent_cases_excluded
    })

    # Optional: Abbruch, wenn keine Fälle mehr übrig sind
    if num_complete_cases == 0:
        print(f"Bei k={k} sind keine Fälle mehr übrig. Analyse gestoppt.")
        break

# Ergebnisse in einen DataFrame umwandeln
tradeoff_df = pd.DataFrame(analysis_results)
tradeoff_df.to_csv(
    get_now_label() + 'tradeoff_table_7_Tage.csv',
    index=False,
)

# --- 5. Ergebnis anzeigen ---

print("\n--- Trade-off-Analyse-Tabelle ---")
# Zeigt die Spalten, die für die Entscheidung wichtig sind
print(
    tradeoff_df[['k_num_params', 'num_cases_kept', 'percent_cases_kept', 'percent_cases_excluded', 'last_param_added']])

# --- 6. (Optional) Visualisierung ---
# Führen Sie dies in einer Umgebung wie Jupyter Notebook aus:
# import matplotlib.pyplot as plt
#
# ax = tradeoff_df.plot(
#     x='k_num_params',
#     y='percent_cases_kept',
#     title='Trade-off: Parameter-Anzahl vs. Fall-Abdeckung',
#     grid=True,
#     figsize=(10, 6)
# )
# ax.set_xlabel("Anzahl der Top-Parameter im Kernset (k)")
# ax.set_ylabel("% der Fallprozeduren, die *alle* k Parameter haben")
# plt.show()
