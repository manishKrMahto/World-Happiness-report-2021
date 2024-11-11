import streamlit as st 
import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('world-happiness-report-2021.csv')
encoder = pickle.load(open('encoder.pkl' , 'rb'))
lr = pickle.load(open('lr.pkl' , 'rb'))

st.title('World Happiness Score Prediction')

regional_indicator = list(df['Regional indicator'].unique())

selected_region = st.selectbox('choose region' , list(df['Regional indicator'].unique()))
GDP_per_capita = st.slider('Choose GDP per capita from (6 to 12 )', min_value = 6.0 , max_value = 12.0 , step = 0.01)
social_support = st.slider('Choose Social support from (0 to 1 )', min_value = 0.0 , max_value = 1.0 , step = 0.01)
healthy_life_expectancy = st.slider('Choose Healthy life expectancy from (40 to 80 )', min_value = 40.0 , max_value = 80.0 , step = 0.01)
freedom_score = st.slider('Choose Freedom to make life choice from (0 to 1 )', min_value = 0.0 , max_value = 1.0 , step = 0.01)
generosity = st.slider('Choose Generosity from (-0.3 to 0.6 )', min_value = -0.3 , max_value = 0.6 , step = 0.01)


input_list = [GDP_per_capita,social_support,healthy_life_expectancy,freedom_score,generosity]

input_array = np.array(input_list)

encoded_region = encoder.transform(np.array([[selected_region]])).toarray()
input_array = np.hstack((input_array.reshape(1,-1) , encoded_region))

status = st.button('What ?')
if status :
    st.success(f'Ladder score : {lr.predict(input_array)}')