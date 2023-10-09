from fastapi import FastAPI
from starlette.responses import JSONResponse
import pickle
import pandas as pd
from datetime import timedelta
import datetime

app = FastAPI()

with open('../models/forecasting/final_prophet_model.p', 'rb') as prophet_model_file:
    prophet_model = pickle.load(prophet_model_file)
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
    return """
    This project is for creating a machine learning model and forecasting model to predict the revenue for the American retailer that has 10 stores across 3 different states: California (CA), Texas (TX), and Wisconsin (WI).

    Two different types of problems were identified and solved using the models for this project.

    There are 4 endpoints created for this project:

    - / (GET): First endpoint to reach this page.

    - /health/ (GET): Second endpoint for a welcoming message.

    - /sales/national/ (GET): Third endpoint for getting the prediction of forecasting model. The model expects date in string format.

    - /sales/stores/items/ (GET): Fourth endpoint for the predictive model. The model expects 3 parameters as input: item id, store id, and date for prediction.
    
    Link to Github Repository: https://github.com/ronik999/Machine_learning_as_a_service/tree/master
    
    API URL: https://protected-lake-95023-3e1d8126370a.herokuapp.com/
    """
@app.get('/health', status_code=200)
def healthcheck():
    return 'Get ready to predict the item revenue or forecast if you want!!!'


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


