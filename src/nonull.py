import pandas as pd


train_data = pd.read_csv('../results/cases_train_processed.csv', parse_dates=['date_confirmation'])

for i in train_data[train_data.date_confirmation.isna()].index:
   date = train_data[train_data.Combined_Key == train_data.iloc[i].Combined_Key].date_confirmation.mean()
   train_data.at[i, 'date_confirmation'] = date

train_data.to_csv('../results/cases_train_processed.csv', index=False)