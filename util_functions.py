import os

import pandas as pd
import datetime

OUTPUT_FOLDER = 'Outputs/'

def concat_csv_files(
        folder_path,
        csv_dtype,
        csv_cols,
        csv_sep=",",
        csv_encoding="utf-8",
):
    """
    Concats all CSV files in a specified folder into a single pandas DataFrame,
    but only if they all have the exact same columns.

    Args:
        folder_path (str): The path to the folder containing the CSV files.
        csv_dtype (dict or None): A dictionary mapping column names to dtypes.
        csv_cols (list): A list of column names.
        csv_sep (str): The CSV separator to use.
        csv_encoding (str): The CSV encoding.

    Returns:
        pd.DataFrame: A single DataFrame containing all merged data, or None
                      if no CSV files are found.

    Raises:
        ValueError: If the columns of the CSV files are not identical.
    """
    # Create a list to store DataFrames
    dfs = []

    # Get a list of all CSV files in the folder
    print(f"Starting to read csv files in {folder_path}...")
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    # If no CSV files are found, return None
    if not csv_files:
        print(f"No CSV files found in '{folder_path}'.")
        return None

    # Read the first file to establish the reference columns
    first_file = os.path.join(folder_path, csv_files[0])
    try:
        reference_df = pd.read_csv(
            first_file,
            dtype=csv_dtype,
            sep=csv_sep,
            encoding=csv_encoding,
            quotechar='"',
            # on_bad_lines="warn",
            keep_default_na=False,
            usecols=csv_cols,
        )
        reference_cols = list(reference_df.columns)
        dfs.append(reference_df)
        print(f"Successfully read '{first_file}'.")
    except Exception as e:
        print(f"Error reading first file '{first_file}': {e}")
        return None

    # Iterate through the rest of the files
    for csv_file in csv_files[1:]:
        file_path = os.path.join(folder_path, csv_file)
        try:
            current_df = pd.read_csv(
                file_path,
                dtype=csv_dtype,
                sep=csv_sep,
                encoding=csv_encoding,
                quotechar='"',
                on_bad_lines="warn",
                keep_default_na=False,
                usecols=csv_cols,
            )
            current_cols = list(current_df.columns)

            # Check if columns are exactly the same as the reference
            if current_cols != reference_cols:
                raise ValueError(
                    f"""Column mismatch! '{csv_file}' has columns {current_cols},
                    but expected columns are {reference_cols}."""
                )

            # If columns match, append the DataFrame to the list
            dfs.append(current_df)
            print(f"Successfully read and appended '{file_path}'.")
        except Exception as e:
            print(f"Error processing file '{file_path}': {e}")
            raise  # Re-raise the exception after printing the error

    # Concatenate all DataFrames in the list
    concatted_df = pd.concat(dfs, ignore_index=True)
    return concatted_df

def write_to_file(df, fn_out):
    filename = OUTPUT_FOLDER + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M") + "_" + fn_out
    df.to_csv(
        filename,
        single_file=True,
        index=False,
    )

def add_numeric_columns(ddf, columns_labels_original):
    for column in columns_labels_original:
        column_label_new = column + '_num'
        ddf[column_label_new] = pd.to_numeric(ddf[column].str.replace(',', '.'), errors='coerce')
    return ddf