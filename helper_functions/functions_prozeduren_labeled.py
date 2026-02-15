import pandas as pd

from helper_functions.util_functions import concat_csv_files
from helper_functions.config import LABEL_CONFIDENCE, PATH_TO_LABELED_CT_REPORTS


def get_labeled_prozeduren_from_file():
    df_befunde_labeled = concat_csv_files(
        folder_path=PATH_TO_LABELED_CT_REPORTS,
        csv_dtype=None,
        csv_cols=[
            'Fallnummer',
            'prozedur_datetime',
            'Patientennummer',
            'geschlecht',
            'alter_bei_prozedur',
            'predicted_label',
            'confidence'
        ]
    )

    df_befunde_labeled_dedup = df_befunde_labeled.drop_duplicates().copy()
    unique_patients = df_befunde_labeled_dedup['Patientennummer'].unique()

    print(f"Insgesamt {len(df_befunde_labeled_dedup)} Untersuchungsereignisse von {len(unique_patients)} Patienten.")
    # print(df_befunde_labeled_dedup['geschlecht'].value_counts())
    # print(df_befunde_labeled_dedup['geschlecht'].value_counts(normalize=True))
    # print(df_befunde_labeled_dedup['predicted_label'].value_counts())
    # print(df_befunde_labeled_dedup['predicted_label'].value_counts(normalize=True))

    df_befunde_labeled['prozedur_datetime'] = pd.to_datetime(df_befunde_labeled['prozedur_datetime'])
    # df_befunde_labeled['geschlecht'] = df_befunde_labeled['geschlecht'].astype('category')
    df_befunde_labeled['alter_bei_prozedur'] = df_befunde_labeled['alter_bei_prozedur'].astype(int)
    # df_befunde_labeled['predicted_label'] = df_befunde_labeled['predicted_label'].astype('category')

    return df_befunde_labeled

def get_prozeduren_for_training():
    df_prozeduren = get_labeled_prozeduren_from_file()
    # df_prozeduren["conf_gte_ninety"] = df_prozeduren['confidence'] >= 0.9
    # print(df_prozeduren[["predicted_label", "conf_gte_ninety"]].value_counts(sort=False))
    df_prozeduren_dedup = df_prozeduren.drop_duplicates().copy()
    # df_prozeduren_dedup["conf_gte_ninety"] = df_prozeduren_dedup['confidence'] >= 0.9
    # print(df_prozeduren_dedup[["predicted_label", "conf_gte_ninety"]].value_counts(sort=False))

    # 1.1 Filtere auf LAE-Kategorie 0 (=LAE ausgeschlossen) und 1 (=LAE nachgewiesen) und confidence >= 0.9
    df_prozeduren_for_training = df_prozeduren_dedup[
        ((df_prozeduren_dedup['predicted_label'] == "Keine LE (0)")
         | (df_prozeduren_dedup['predicted_label'] == "LE vorhanden (1)"))
        & (df_prozeduren_dedup['confidence'] >= LABEL_CONFIDENCE)
        ].copy()


    # print(df_prozeduren_for_training[["predicted_label", "conf_gte_ninety"]].value_counts(sort=False))
    print(f"{len(df_prozeduren_for_training)} von {len(df_prozeduren)} Prozeduren haben ein "
          f"confidence-Wert >= {LABEL_CONFIDENCE} und sind in LAE-Kategorie 'Keine LE (0)' oder 'LE vorhanden (1)'.")

    return df_prozeduren_for_training
