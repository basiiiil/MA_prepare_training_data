from dask import dataframe as dd
import numpy as np
from main import get_labelled_prozeduren, get_latest_lab_values, get_time_window_table, get_now_label
from merge_data import add_laborwerte_to_prozeduren
import datetime

print(datetime.datetime.now().strftime("%H:%M:%S") + " - Let's go!")
ddf = dd.read_csv(
    '2025-10-20_09:17_prozeduren_with_latest_lab_values.csv',
    blocksize='64MB',
)
ddf_dedup_cases = ddf.drop_duplicates(subset=['Fallnummer', 'prozedur_datetime']).copy()
num_cases = len(ddf_dedup_cases)
print(f"Anzahl Prozeduren: {num_cases}")



zeitfenster_pivot = get_time_window_table(ddf, 24)
zeitfenster_pivot = zeitfenster_pivot.reset_index().rename_axis(None, axis=1)

for i in range(7):
    cols_to_sum = list(range(i+1))
    print(cols_to_sum)
    zeitfenster_pivot[f'abdeckung_in_{(i+1)*24}h'] = zeitfenster_pivot.apply(
        lambda row: np.round(row[cols_to_sum].sum() * 100 / num_cases, 1), axis=1
    )


zeitfenster_pivot.to_csv(
    get_now_label() + 'laborwerte_tage_abdeckung.csv',
    index=False,
)



# zeitfenster_pivot['abdeckung'] = np.round((zeitfenster_pivot['anzahl_einträge'] * 100 / num_cases), 1)
# zeitfenster_labels = ['parameterid_effektiv']
# zeitfenster_labels.extend([f'{i * 8}-{(i + 1) * 8}h' for i in range(21)])
# zeitfenster_labels.extend(['anzahl_einträge', 'abdeckung'])
# zeitfenster_pivot.columns = zeitfenster_labels



