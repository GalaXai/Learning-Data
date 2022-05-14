import streamlit as st
import pickle
import numpy as np
import pandas as pd

df=pd.read_csv('df.csv')

pipe = pickle.load(open("pipe.pkl", "rb"))

st.title('Laptop Price Predictor')

#Brand
company = st.selectbox('Brand',df['Company'].unique())

#Type of Laptop
laptop_type = st.selectbox("Type", df['TypeName'].unique())

#Ram
ram = st.selectbox("Ram", df['Ram'].sort_values(ascending=False).unique())

#Weight
weight = st.selectbox("Weight", df['Weight'].sort_values(ascending=False).unique())

#TouchScreen
touchscreen = st.selectbox("TouchScreen", ['No', 'Yes'])

#IPS
ips = st.selectbox("IPS", ['No', 'Yes'])

#Screen Size
screen_size = st.number_input('Screen Size (in Inches)')

#Resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#Specs
cpu= st.selectbox('CPU',df['Cpu_brand'].unique())
hdd= st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
ssd= st.selectbox('SSD(in GB)',[0,128,256,512,1024])
gpu = st.selectbox('GPU',df['Gpu_brand'].unique())
os = st.selectbox('OS',df['Os'].unique())

# Prediction

if st.button('Predict Price'):
    ppi = None
    if touchscreen == "Yes":
        touchscreen = 1
    else:
        touchscreen = 0
    if ips == "Yes":
        ips = 1
    else:
        ips = 0
    
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res **2) + (Y_res **2)) / screen_size

    query = np.array([company,laptop_type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    query = query.reshape(1,12)
    prediction = str(int(np.exp(pipe.predict(query)[0])))
    st.title("The predicted price of the configuartion is " + prediction)
