from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import timedelta
import pandas as pd
import joblib
import numpy as np
from starlette.responses import JSONResponse


def get_scores(y_true, y_pred, split):
    '''
    FUNCTION TO GET THE SCORES FOR REGRESSION
    '''
    mae = mean_absolute_error(y_true, y_pred)
    r2_scores = r2_score(y_true,y_pred)
    mse = mean_squared_error(y_true, y_pred, squared=False)
    print(f'MAE SCORE for {split}:{mae}')
    print(f'RMSE SCORE for {split}:{mse}')
    print(f'R2 SCORE for {split}:{r2_scores}')


def predict_prophet_model(model, df_test, ):
    future_dates = model.make_future_dataframe(periods=7)
    prediction = model.predict(future_dates)
    df_test = df_test.rename(columns={'date': 'ds', 'revenue': 'y'})
    compare_table = df_test.merge(prediction, on='ds', how='left')
    get_scores(compare_table['y'], compare_table['yhat'], split="TEST")
    model.plot(prediction)
    model.plot_components(prediction)

    return prediction


def predict_prophet_inference(model_name_file, date):

    start_date = pd.to_datetime(date)
    end_date = start_date + timedelta(days=6)

    future_df = pd.DataFrame({'ds': pd.date_range(start=start_date, end=end_date, freq='D')})
    with open('../../models/forecasting/'+str(model_name_file), 'rb') as model_file:
        model = joblib.load(model_file)
    prediction = model.predict(future_df)
    prediction = prediction[['ds', 'yhat']]
    prediction['ds'] = prediction['ds'].astype(str)

    return prediction.values.tolist()


def predict_naive_model_scores(y_test):
    '''
    Model to get naive scores:
    y_test: Add test values
    '''
    y_test['naive_score'] = np.mean(y_test['revenue'])
    get_scores(y_test['revenue'], y_test['naive_score'], split='Naive Model')




