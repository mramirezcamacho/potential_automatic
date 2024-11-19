import os
import pandas as pd
import numpy as np

lastDocColumns = ['*shopID', 'Operations Category (Grocery)', 'Block orders in Cash?',
                  "If the store accepts cash payment by couriers?", 'Suspend POS Terminal Orders?', 'Payment setting for shopper mode stores', 'Picking Mode', 'Leads Potential']

important_columns = ['country_code', 'shop_id', 'potential', 'new_potential']


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
    df_global = pd.read_csv(f'data/{get_most_recent_csv('data')}')
    columns_to_check = ['shop_id', 'new_potential']

    df_global['potential'] = df_global['potential'].apply(
        lambda x: 'no_data' if pd.isna(x) or x == '' else x)
    df_global['shop_id'] = df_global['shop_id'].apply(
        lambda x: str(x))
    nan_condition = df_global[columns_to_check].isnull()
    empty_string_condition = df_global[columns_to_check] == ''
    combined_condition = nan_condition | empty_string_condition
    rows_to_drop = combined_condition.any(axis=1)
    df_global = df_global[~rows_to_drop]
    df_global.reset_index(drop=True, inplace=True)

    df_global['has_to_change'] = df_global['potential'] != df_global['new_potential']

    paises = df_global['country_code'].unique()
    data_per_country = {}
    for pais in paises:
        data_to_insert = df_global[df_global['country_code'] == pais]
        data_to_insert = data_to_insert[data_to_insert['has_to_change'] == True]
        print(data_to_insert)
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

    for file_name, df_small in for_production.items():
        df_small.to_excel(
            f'files_to_upload/{file_name}.xlsx', index=False, engine='openpyxl')


if __name__ == '__main__':
    divide_per_country()
