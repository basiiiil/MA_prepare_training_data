import numpy as np
import pandas as pd


def get_stammdaten_inpatients_df():
    """
    :return: pandas dataframe
    """
    df_source = pd.read_csv(
        'fromDIZ/Stammdaten/24.07.2025_Patientenliste_3-222_Stammdaten_sf.csv',
        dtype={
            'Fallnummer': np.int_,
            # 'Patientennummer': np.int_,
            'Alter': np.int_,
            # 'Aufnahmedatum': np.str_,
        },
        usecols=[
            'Fallnummer',
            'Patientennummer',
            'Geschlecht',
            'Alter',
            'Bewegung Behandlungsart',
            # 'Aufnahmedatum'
        ]
    )

    df_source_nachgereicht = pd.read_csv(
        'fromDIZ/Stammdaten/Stammdaten.csv',
        dtype={
            # 'Patient': np.int_,
            'Fall': np.int_,
            'GebDatum': np.str_
        },
        usecols=[
            'Fall',
            'Patient',
            'G',
            'GebDatum',
            'Fa'
        ]
    )

    df_source['behandlungsart'] = df_source['Bewegung Behandlungsart']
    df_source['geschlecht'] = df_source['Geschlecht']
    df_source['alter_bei_prozedur'] = df_source['Alter']

    df_source_nachgereicht['Fallnummer'] = df_source_nachgereicht['Fall']
    df_source_nachgereicht['Patientennummer'] = df_source_nachgereicht['Patient']
    df_source_nachgereicht['behandlungsart'] = df_source_nachgereicht['Fa']
    df_source_nachgereicht['geschlecht'] = df_source_nachgereicht['G']
    df_source_nachgereicht['geburtsdatum'] = pd.to_datetime(
        df_source_nachgereicht['GebDatum'], format='%Y-%m-%d'
    )

    # 3. Merge all Fallnummern and drop duplicates
    df_stammdaten_all = pd.concat(
        [
            df_source[
                [
                    'Fallnummer',
                    'Patientennummer',
                    'geschlecht',
                    # 'geburtsdatum',
                    'alter_bei_prozedur',
                    'behandlungsart'
                ]
            ],
            df_source_nachgereicht[
                [
                    'Fallnummer',
                    'Patientennummer',
                    'geschlecht',
                    'geburtsdatum',
                    # 'alter_bei_prozedur',
                    'behandlungsart'
                ]
            ],
        ]
    )
    df_stammdaten_all_inpatients = df_stammdaten_all[
        df_stammdaten_all['behandlungsart'] == 'Stationär'
    ].reset_index(drop=True)
    df_stammdaten_dedup = df_stammdaten_all_inpatients.drop_duplicates().copy()
    num_patients = df_stammdaten_dedup['Patientennummer'].nunique()
    print(f"Es existieren Stammdaten zu {len(df_stammdaten_dedup)} "
          f"stationären Fällen von {num_patients} Patientennummern.")

    return df_stammdaten_dedup

