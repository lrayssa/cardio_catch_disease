import pickle
import pandas as pd


age_scaler = pickle.load(open('/home/lrayssa/Documents/data_science/pa01/cardio_catch_disease/parameters/age_scaler.pkl', 'rb'))
height_scaler = pickle.load(open('/home/lrayssa/Documents/data_science/pa01/cardio_catch_disease/parameters/height_scaler.pkl', 'rb'))
weight_scaler = pickle.load(open('/home/lrayssa/Documents/data_science/pa01/cardio_catch_disease/parameters/weight_scaler.pkl', 'rb'))
ap_hi_scaler = pickle.load(open('/home/lrayssa/Documents/data_science/pa01/cardio_catch_disease/parameters/ap_hi_scaler.pkl', 'rb'))
ap_lo_scaler = pickle.load(open('/home/lrayssa/Documents/data_science/pa01/cardio_catch_disease/parameters/ap_lo_scaler.pkl', 'rb'))
bmi_scaler = pickle.load(open('/home/lrayssa/Documents/data_science/pa01/cardio_catch_disease/parameters/bmi_scaler.pkl', 'rb'))

def feature_engineering( data ):
    data['bmi'] = data.apply(lambda x: x['weight'] / ((x['height'] / 100) ** 2), axis=1)
    data['obesity'] = data['bmi'].apply(lambda x: 3 if x >= 40 else 2 if x >= 30
    else 1 if x >= 25 else 0)

    return data

def data_preparation(data):
    # age
    data['age'] = age_scaler.transform(data[['age']].values)

    # height
    data['height'] = height_scaler.transform(data[['height']].values)

    # weight
    data['weight'] = weight_scaler.transform(data[['weight']].values)

    # ap_hi
    data['ap_hi'] = ap_hi_scaler.transform(data[['ap_hi']].values)

    # ap_lo
    data['ap_lo'] = ap_lo_scaler.transform(data[['ap_lo']].values)

    # bmi
    data['bmi'] = bmi_scaler.transform(data[['bmi']].values)

    cols_selected = ['age', 'ap_hi', 'bmi']

    return data[cols_selected]


def get_prediction(model, data):
    pred = model.predict(data)
    pred_proba = round(model.predict_proba(data).tolist()[0][1] * 100, 2)

    return pred_proba

