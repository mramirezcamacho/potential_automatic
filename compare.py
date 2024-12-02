import pandas as pd
prediction_file = 'NewRankDataset_cristobalnavarro_20241129002848_prediction'
current_file = 'current_potential2'

dtype_dict = {'shop_id': str, 'city_name': str, 'country_code': str}
prediction_df = pd.read_csv(f'ML/predictions/{prediction_file}.csv', encoding='unicode_escape',
                            low_memory=False, dtype=dtype_dict)
dtype_dict = {'shop_id': str, 'city_name': str,
              'country_code': str, 'current_potential': str}
current_df = pd.read_csv(f'ML/predictions/{current_file}.csv', encoding='unicode_escape',
                         low_memory=False, dtype=dtype_dict)
current_df = current_df.fillna('no data')

merged_df = pd.merge(prediction_df, current_df, on='shop_id', how='left')
merged_df['predicted'] = merged_df['new_rank']
merged_df['country_code'] = merged_df['country_code_x']
merged_df = merged_df[['country_code', 'city',
                       'shop_id', 'predicted', 'current_potential']]

for country in merged_df['country_code'].unique():
    print(f'In {country}:')
    for potential_predicted in sorted(list(merged_df[merged_df['country_code'] == country]['predicted'].unique())):
        important_df = merged_df[merged_df['country_code'] == country]
        important_df = important_df[important_df['predicted']
                                    == potential_predicted]
        print(f'the Rs with potential {potential_predicted}:')
        important_df['current_potential'] = important_df['current_potential'].fillna(
            'Churned/No orders')
        values = list(important_df['current_potential'].values)
        for current in sorted(list(set(values))):
            print(f'''Currently {
                  round(100*(values.count(current)/len(values)), 2)}% are {current}''')
