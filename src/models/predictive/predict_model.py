from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pickle
import pandas as pd
import datetime
from src.visualization.visualize import get_actual_vs_predicted_plot
import joblib

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


def predict_model(trained_model, X_train, y_train, X_test, y_test, save_model, save_model_file_name, plot):
    '''
    Input Parameters:
    -----------------
    trained_model: Add Trained Model
    X_train: Splitted train features for predictors (X)
    y_train: Splitted train revenues (y)
    X_test: Splitted test features for predictors (X)
    y_test: Splitted test revenues (y)
    save_model: Bool values to save model or not
    save_model_file_name: Name for saving the model

    Output:
    ________
    model: Trained model

    '''
    y_pred = trained_model.predict(X_test)
    y_pred_train = trained_model.predict(X_train)
    get_scores(y_train, y_pred_train, split='Train')
    get_scores(y_test, y_pred, split="Test")
    if save_model:
        joblib.dump(trained_model, open("../../models/predictive/"+str(save_model_file_name), "wb"))
    if plot:
        get_actual_vs_predicted_plot(y_test, y_pred)

def predict_naive_model_scores(y_test):
    '''
    Model to get naive scores:
    y_test: Add test values
    '''
    mean_prediction = np.mean(y_test)
    naive_pred = np.full_like(y_test, fill_value=mean_prediction)
    get_scores(y_test, naive_pred, split='Naive Model')



def predict_inference_model(store_id, item_id, date):
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

    prediction_df = pd.DataFrame({'item_id':[item_id], 'store_id':[store_id], 'date':[date]})
    prediction_df['date'] = pd.to_datetime(prediction_df['date'])
    prediction_df['cat_id'] = prediction_df['item_id'].str.split('_').str[0]
    prediction_df['state_id'] = prediction_df['store_id'].str.split('_').str[0]
    prediction_df['day'] = prediction_df['date'].dt.day_name()
    prediction_df['month'] = prediction_df['date'].dt.month_name()
    with open('../../models/ord_enc.p', 'rb') as enc_file:
        enc = pickle.load(enc_file)
    col = ['item_id', 'cat_id', 'store_id','state_id', 'day', 'month']
    prediction_df[col] = enc.transform(prediction_df[col])
    prediction_df['date'] = prediction_df['date'].map(datetime.datetime.toordinal)
    prediction_df = prediction_df[['item_id', 'cat_id', 'store_id', 'state_id', 'date', 'day', 'month']]
    with open('../../models/predictive/decision_tree_final_model.p', 'rb') as model_file:
        model = pickle.load(model_file)
    prediction = model.predict(prediction_df)

    return prediction.tolist()