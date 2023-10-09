from prophet import Prophet
import pickle


def train_prophet_model(train_df, model_save, model_name):

    train_df = train_df.rename(columns={'date':'ds', 'revenue':'y'})
    model = Prophet()
    model.add_country_holidays(country_name='USA')
    model.fit(train_df)
    if model_save:
        pickle.dump(model, open("../../models/forecasting/"+str(model_name), "wb"))

    return model

