import os
import pandas as pd
import numpy as np
import warnings

# Filter out FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning,
                        module="numpy._core.fromnumeric")

lastDocColumns = pd.read_excel(
    "gattaran_files/template/template.xlsx").columns.tolist()


def delete_all_files(folder_path):
    try:
        # List all files in the folder
        files = os.listdir(folder_path)

        # Loop through the files and delete them
        for file in files:
            file_path = os.path.join(folder_path, file)

            # Only delete files (ignore subdirectories)
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                print(f"Skipped (not a file): {file_path}")

    except Exception as e:
        print(f"Error: {e}")


def get_most_recent_csv(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError(f"The path '{folder_path}' is not a valid directory.")

    csv_files = [
        os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')
    ]

    if not csv_files:
        raise ValueError

    most_recent_file = max(csv_files, key=os.path.getctime)

    return os.path.basename(most_recent_file)


def divide_per_country():
    df_global = pd.read_csv(
        f'gattaran_files/data_new_old_priority/{get_most_recent_csv('gattaran_files/data_new_old_priority')}')

    df_global['shop_id'] = df_global['shop_id'].apply(
        lambda x: str(x))

    paises = df_global['country_code'].unique()
    data_per_country = {}
    for pais in paises:
        data_to_insert = df_global[df_global['country_code'] == pais]
        data_per_country[pais] = data_to_insert

    # Create a new dictionary to store the split DataFrames
    split_data_per_country = {}

    for pais, df_small in data_per_country.items():
        if len(df_small) > 950:
            split_dfs = np.array_split(
                df_small, range(950, len(df_small), 950))

            for i, chunk in enumerate(split_dfs):
                split_data_per_country[f"{pais}_part_{i+1}"] = chunk
        else:
            split_data_per_country[pais] = df_small

    for_production = {}
    for file_name, df_small in split_data_per_country.items():
        newDF = pd.DataFrame(columns=lastDocColumns)
        for i, row in df_small.iterrows():
            newDF = newDF._append({'*shopID': str(row['shop_id']),
                                   'Leads Potential': row['new_potential']},
                                  ignore_index=True)
        for_production[file_name] = newDF

    delete_all_files("gattaran_files/files_to_upload")

    for file_name, df_small in for_production.items():
        df_small.to_excel(
            f'gattaran_files/files_to_upload/{file_name}.xlsx', index=False, engine='openpyxl')


if __name__ == '__main__':
    divide_per_country()
