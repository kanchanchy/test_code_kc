import pandas as pd
from datetime import datetime

secondsInADay = 86400

# Create a mapping from weekday names to numeric values
weekday_mapping = {
    'Sunday': 0,
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6
}

#file_transaction = "financial_transactions_small.csv"
#file_customer = "customer.csv"
file_order = "data/order.csv"
file_store = "data/store_dept.csv"

df_order = pd.read_csv(file_order)
#df_order = df_order.drop(["weekday", "store", "trip_type"], axis=1)
df_order['weekday'] = df_order['weekday'].map(weekday_mapping)
df_order['date'] = pd.to_datetime(df_order['date'], format='%Y-%m-%d', errors='coerce')
df_order = df_order.dropna(subset=['date'])
df_order['date'] = df_order['date'].astype('int64') // ((10**9)*secondsInADay)
df_order.rename(columns={'store': 'store_id'}, inplace=True)

df_order = pd.get_dummies(df_order, columns=['weekday'])

df_store = pd.read_csv(file_store)

df_final = pd.merge(df_order, df_store, left_on='store_id', right_on='store', how='inner')
df_final = df_final.drop(["store_id"], axis=1)
cols = df_final.columns.tolist()
new_order = ['o_order_id'] + [col for col in cols if col not in ['o_order_id', 'trip_type']] + ['trip_type']
df_final = df_final[new_order]
df_final.to_csv('data/processed_order_store.csv', index=False)
print(df_final.head())
print(df_final.columns.tolist())

#print(df_order['date'].max())





