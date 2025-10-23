from main import get_labelled_prozeduren
from merge_data import get_prozedur_charlson_pivot

df_prozeduren = get_labelled_prozeduren(168)
df = get_prozedur_charlson_pivot(df_prozeduren)