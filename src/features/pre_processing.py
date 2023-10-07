import pandas as pd


def reduce_dtype(df):
    int64_columns = df.select_dtypes(include=['int64']).columns
    float64_columns = df.select_dtypes(include=['float64']).columns
    df[int64_columns] = df[int64_columns].astype('int16')
    df[float64_columns] = df[float64_columns].astype('float32')
    int32_columns = df.select_dtypes(include=['int32']).columns
    df[int32_columns] = df[int32_columns].astype('int16')

    return df


def events(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.groupby('date').agg({'event_name': ', '.join, 'event_type': ', '.join}).reset_index()
    df[['event_type_1', 'event_type_2']] = df['event_type'].str.split(',', expand=True)
    df[['event_name_1', 'event_name_2']] = df['event_name'].str.split(',', expand=True)
    df['event_type_1'] = df['event_type_1'].str.strip()
    df['event_type_2'] = df['event_type_2'].str.strip()
    return df.drop(['event_type', 'event_name'], axis=1)



def date_features_forecasting(calendar, calendar_events):
    calendar['date'] = pd.to_datetime(calendar['date'])
    calendar['day'] = calendar['date'].dt.day_name()
    calendar['month'] = calendar['date'].dt.month_name()
    calendar_events['date'] = pd.to_datetime(calendar_events['date'])
    calendar = calendar.merge(calendar_events, on='date', how='left')

    return calendar


def date_features(calendar):
    calendar['date'] = pd.to_datetime(calendar['date'])
    calendar['day'] = calendar['date'].dt.day_name()
    calendar['month'] = calendar['date'].dt.month_name()

    return calendar


def prepare_forecasting_train_data(df, df_calendar, df_calendar_events, df_weekly_sales):
    df = pd.melt(df, id_vars=df.columns[:6],
                 value_vars=df.columns[6:], var_name='d', value_name='sales')
    df['sales'] = df['sales'].astype('int16')
    print("MELTED")
    df_calendar = date_features_forecasting(df_calendar, df_calendar_events)
    print("Adding Calendar events")
    df = df.merge(df_calendar, on=['d'], how='left')
    print("ADDING WEEKLY SALES")
    df = df.merge(df_weekly_sales, on=['wm_yr_wk', 'store_id', 'item_id'], how='left')
    print("CALCULATING REVENUE")
    df['revenue'] = df['sales'] * df['sell_price']
    print("DROPPING COLUMNS")
    df = df.drop(['id', 'sales', 'd', 'wm_yr_wk', 'sell_price'], axis=1)

    return df

def melt_df(df):

    return pd.melt(df, id_vars=df.columns[:6],
                 value_vars=df.columns[6:], var_name='d', value_name='sales')



