import streamlit as st
import datetime
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


def get_individual_data():
    st.text('Here you can input your data and get the prediction of the probability of having a cardiovascular disease')

    name = st.text_input(label='What is your name?')

    # birth date (to calculate the age in days): date input
    date_of_birth = st.date_input(label='Date of birth', min_value=datetime.date(1900, 1, 1), max_value=datetime.date.today())
    age = (datetime.date.today() - date_of_birth).days

    # gender: radio box or selectbox

    gender = st.selectbox(label='Choose your gender:', options=('female', 'male'))


    # height: number input
    height = st.number_input(label='Whats your height in centimeters?', min_value=54.0, max_value=215.0,
                             step=0.5, value=165.0)

    # current weight: number input
    weight = st.number_input(label='Whats your weight in kg?', min_value=30.0, max_value=350.0, step=0.5, value=72.0)

    # systolic blood pressure: number input
    ap_hi = st.number_input(label='Input your systolic blood pressure:', min_value=90, max_value=170, step=1, value=120)

    # diastolic blood pressure: number input
    ap_lo = st.number_input(label='Input your diastolic blood pressure:', min_value=65, max_value=105, step=1, value=80)

    # cholesterol : select slider
    cholesterol = st.select_slider(label='How are your cholesterol levels?',
                                   options=['normal', 'above normal', 'well_above_normal'], value='normal')
    # glucose: select slider
    glucose = st.select_slider(label='How are your glucose levels?',
                               options=['normal', 'above normal', 'well_above_normal'], value='normal')

    # smoking
    smoke = st.radio(label='Do you smoke?', options=['Yes', 'No'])

    # alcohol
    alcohol_intake = st.radio(label='Do you drink alcohol frequently?', options=['Yes', 'No'])

    # phisical activity: radio or selectbox
    physycal_activity = st.radio(label='Do you execise frequently?', options=['Yes', 'No'])

    make_prediction = st.button(label='Predict')

    features = [age, gender, height, weight, ap_hi, ap_lo, cholesterol, glucose, smoke,
                 alcohol_intake, physycal_activity]
    data = pd.DataFrame(features).T
    data.columns = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                        'cholesterol', 'gluc', 'smoke', 'alco', 'active']

    return data, name, make_prediction

def data_transformation(data):
    # ------------ encoding ------------
    # gender to label
    data['gender'] = data['gender'].apply(lambda x: 1 if x == 'female' else 2)

    # cholesterol and gluc to ordinal
    ordinal = ['cholesterol', 'gluc']
    for i in ordinal:
        data[i] = data[i].apply(lambda x: 1 if x == 'normal' else 2 if x == 'above normal' else 3)

    # smoke, alco and active to label
    label = ['smoke', 'alco', 'active']
    for i in label:
        data[i] = data[i].apply(lambda x: 1 if x == 'Yes' else 0)

    # ------------- change dtypes --------------
    data[['age', 'height', 'ap_hi', 'ap_lo']] = data[['age', 'height', 'ap_hi', 'ap_lo']] .astype('int64')
    data['weight'] = data['weight'].astype('float64')

    # ----------- calling database ----------------
    database = pd.read_csv('/home/lrayssa/Documents/data_science/pa01/cardio_catch_disease/data/test.csv')
    database = database.drop(database.columns[0], axis=1)

    data = pd.concat([data, database]).reset_index(drop=True)

    return(data)


def data_preparation(data):

    # -------------- feature engineering -------------
    data['bmi'] = data.apply(lambda x: bmi(x['height'], x['weight']), axis=1)

    # -------------- data preparation ---------------
    # age
    data['age'] = mms.fit_transform(data[['age']].values)

    # height
    data['height'] = mms.fit_transform(data[['height']].values)

    # weight
    data['weight'] = mms.fit_transform(data[['weight']].values)

    # ap_hi
    data['ap_hi'] = mms.fit_transform(data[['ap_hi']].values)

    # ap_lo
    data['ap_lo'] = mms.fit_transform(data[['ap_lo']].values)

    # bmi
    data['bmi'] = mms.fit_transform(data[['bmi']].values)

    cols_selected = ['age', 'ap_hi', 'bmi']

    return data[cols_selected]

def prediction(data, model):

    yhat = model.predict(data)
    yhat_proba = model.predict_proba(data).tolist()

    yhat_individual = yhat[0]
    yhat_proba_individual = round(yhat_proba[0][1]*100,2)

    return yhat_proba_individual


model = pickle.load(open( '/home/lrayssa/Documents/data_science/pa01/cardio_catch_disease/models/gb_model.pkl', 'rb' ))
mms = MinMaxScaler()


def bmi(height, weight):
    bmi = bmi = weight / ((height / 100) ** 2)

    return bmi


st.title( 'Cardio Catch Disease' )
st.markdown('Welcome to the best app to predict if you have a cardiovascular disease or not!')


st.sidebar.title('Menu')

page = st.sidebar.radio('Choose an option', options=['Single prediction', 'Model', 'Recomendations'])

if __name__ == '__main__':
    if page == 'Single prediction':
        individual_data =get_individual_data()
        data = individual_data[0]
        name = individual_data[1]
        predict = individual_data[2]

        data = data_transformation(data)

        data = data_preparation(data)


        yhat = prediction(data, model)

        if predict == True:
            st.write('{}, there are {}% chance of you having a cardiovascular disease'.format(name, yhat))


    elif page == 'Model':
        st.text('Model details here!  -- in construction (readme)')

    else:
        st.text('Here are our recommendations from next models! -- in construction (business presentation)')