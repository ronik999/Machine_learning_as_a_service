from fastapi import FastAPI
import joblib
from starlette.responses import JSONResponse
import pickle
import pandas as pd
from datetime import timedelta
import datetime

app = FastAPI()

with open('../models/forecasting/final_prophet_model.p', 'rb') as prophet_model_file:
    prophet_model = joblib.load(prophet_model_file)
with open('../models/predictive/decision_tree_final_model.p', 'rb') as tree_model_file:
    tree_model = pickle.load(tree_model_file)
with open('../models/ord_enc.p', 'rb') as enc_file:
    ord_enc = pickle.load(enc_file)


def predict_prophet_inference(model, date):

    start_date = pd.to_datetime(date)
    end_date = start_date + timedelta(days=6)

    future_df = pd.DataFrame({'ds': pd.date_range(start=start_date, end=end_date, freq='D')})
    prediction = model.predict(future_df)
    prediction = prediction[['ds', 'yhat']]
    prediction['ds'] = prediction['ds'].astype(str)

    return JSONResponse(prediction.values.tolist())


def predict_predictive_inference_model(prediction_df, model, enc):
    '''
    INPUT
    _____
    store_id: Add store ID
    item_id: Add item ID
    date: Date for prediction

    OUTPUT:
    -------
    prediction: Return predicted value accroding to store ID and item ID
    '''

    prediction_df['date'] = pd.to_datetime(prediction_df['date'])
    prediction_df['cat_id'] = prediction_df['item_id'].str.split('_').str[0]
    prediction_df['state_id'] = prediction_df['store_id'].str.split('_').str[0]
    prediction_df['day'] = prediction_df['date'].dt.day_name()
    prediction_df['month'] = prediction_df['date'].dt.month_name()
    col = ['item_id', 'cat_id', 'store_id','state_id', 'day', 'month']
    prediction_df[col] = enc.transform(prediction_df[col])
    prediction_df['date'] = prediction_df['date'].map(datetime.datetime.toordinal)
    prediction_df = prediction_df[['item_id', 'cat_id', 'store_id', 'state_id', 'date', 'day', 'month']]
    prediction = model.predict(prediction_df)

    return JSONResponse(prediction.tolist())


def forecast_format_features(date: str):
    return {'date': [date]}


def predictive_format_features(
    date: str,
    item_id: str,
    store_id: str,
    ):
    return {
        'date': [date],
        'item_id':[item_id],
        'store_id': [store_id]
    }


@app.get("/")
def read_root():

    return 'This project is created for predicting and forecasting the revenue of a american retailer'


@app.get('/health', status_code=200)
def healthcheck():
    return 'Get ready to predict and forecast sales!'


@app.get('/sales/national', status_code=200)
def forecast_model(date: str):
    forecast_input = forecast_format_features(date)
    forecast_input = forecast_input['date'][0]
    pred = predict_prophet_inference(prophet_model, forecast_input)
    return pred


@app.get('/sales/stores/items', status_code=200)
def predictive_model(date: str, item_id: str, store_id:str):
    predictive_input = predictive_format_features(date, item_id, store_id)
    predictive_input = pd.DataFrame(predictive_input)
    print(predictive_input)
    pred = predict_predictive_inference_model(predictive_input,
                                              tree_model, ord_enc)
    return pred


