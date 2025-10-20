from main import get_labelled_prozeduren, get_latest_lab_values, get_time_window_table, get_now_label
from merge_data import add_laborwerte_to_prozeduren
import datetime

print(datetime.datetime.now().strftime("%H:%M:%S") + " - Let's go!")
df_prozeduren_final = get_labelled_prozeduren(24 * 7)

# Füge Laborwerte zu Prozeduren hinzu
ddf_prozeduren_mit_labor = add_laborwerte_to_prozeduren(df_prozeduren_final)
ddf_filtered = get_latest_lab_values(ddf_prozeduren_mit_labor)
zeitfenster_pivot = get_time_window_table(ddf_filtered, 8)
zeitfenster_pivot = zeitfenster_pivot.reset_index().rename_axis(None, axis=1)

zeitfenster_pivot.to_csv(
    get_now_label() + 'laborwerte_zeitfenster.csv',
    # single_file=True,
    index=False
)