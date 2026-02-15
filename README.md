## PE risk score - dataset preparation

### What this repo does
This repo prepares the datasets to be used to (1) label the CT reports using medBERT.de and (2) train the final risk prediction models.

There are to parts in this repo:
1. `get_prozeduren_for_labeling.py` contains the code to merge the filtered CT reports with the actual procedures. Its output is a dataset of deduplicated procedures with their respective report texts, to be labeled using medBERT.de.
2. `get_dataset_for_risk_prediction_training.py` contains the code to add all relevant data points (lab results, diagnoses, etc.) to the labeled procedures.

### How to use this repo

1. Put all source files into a `source_data` folder:
   1. the (filtered) radiology reports in `/source_data/ct_reports`,
   2. procedures list in `/source_data/procedures`,
   3. diagnoses in `/source_data/diagnoses`,
   4. lab results in `/source_data/lab_results`,
   5. patient master data in `/source_data/patient_master_data`.
   
   If you use different folder names, specify them in `config.py`.
2. Run `get_prozeduren_for_labeling.py`. Use the output (see `/output/for_labeling`) to run the labeling pipeline using medBERT.de.
3. Put the labeled CT reports inside `/source_data/labeled_ct_reports` (or specify folder name in `config.py`).
4. Run `get_dataset_for_risk_prediction_training.py` to get the final dataset to be used to train the PE risk prediction models. 

NOTE: There are a couple of additional scripts for analysis of different datasets inside `/helper_functions`. They are not well documented, but might still be helpful.