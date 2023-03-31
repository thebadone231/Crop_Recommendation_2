import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np


df = pd.read_csv('Crop_recommendation.csv')
x = df.drop(['label'], axis = 1)
y = df['label']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
type(x_train)
y_train, y_test = np.array(y_train), np.array(y_test)
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse = False)
y1reshaped = y_train.reshape(len(y_train), 1)
Ytr = encoder.fit_transform(y1reshaped)
y2reshaped = y_test.reshape(len(y_test), 1)
Yts = encoder.fit_transform(y2reshaped)
from sklearn.neighbors import KNeighborsClassifier  
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,Ytr)    

st.title('Crop Recommendation Prediction')
st.sidebar.header('Environment Data')



# FUNCTION
def user_report():
  N = st.sidebar.slider('Nitrogen', 0,100, 1 )
  P = st.sidebar.slider('Phosphorus', 0,100, 1 )
  K = st.sidebar.slider('Potassium', 0,100, 1 )
  Temp = st.sidebar.slider('Temperature', 10,45, 1 )
  Hum = st.number_input("Humidity", step = 0.1)
  pH = st.number_input("pH", step = 0.01)
  Rain = st.sidebar.slider('Rainfall', 0,400, 1)


  user_report_data = {
      'N':N,
      'P':P,
      'K':K,
      'temperature':Temp,
      'humidity':Hum,
      'ph':pH,
      'rainfall':Rain,
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

user_data = user_report()
st.header('Environment Data')
st.write(user_data)

crop_predict = model.predict(user_data)
for predict1 in range(22):
  if crop_predict[0,predict1] == 1.0:
    if predict1 == 0:                                                                                              # Above we have converted the crop names into numerical form, so that we can apply the machine learning model easily. Now we have to again change the numerical values into names of crop so that we can print it when required.
      crop_name = 'Apple'
    elif predict1 == 1:
      crop_name = 'Banana'
    elif predict1 == 2:
      crop_name = 'Blackgram'
    elif predict1 == 3:
      crop_name = 'Chickpea'
    elif predict1 == 4:
      crop_name = 'Coconut'
    elif predict1 == 5:
      crop_name = 'Coffee'
    elif predict1 == 6:
      crop_name = 'Cotton'
    elif predict1 == 7:
      crop_name = 'Grapes'
    elif predict1 == 8:
      crop_name = 'Jute'
    elif predict1 == 9:
      crop_name = 'Kidneybeans'
    elif predict1 == 10:
      crop_name = 'Lentil'
    elif predict1 == 11:
      crop_name = 'Maize'
    elif predict1 == 12:
      crop_name = 'Mango'
    elif predict1 == 13:
      crop_name = 'Mothbeans'
    elif predict1 == 14:
      crop_name = 'Mungbeans'
    elif predict1 == 15:
      crop_name = 'Muskmelon'
    elif predict1 == 16:
      crop_name = 'Orange'
    elif predict1 == 17:
      crop_name = 'Papaya'
    elif predict1 == 18:
      crop_name = 'Pigeonpeas'
    elif predict1 == 19:
      crop_name = 'Pomegranate'
    elif predict1 == 20:
      crop_name = 'Rice'
    elif predict1 == 21:
      crop_name = 'Watermelon'
    break
  else:
    crop_name = "No crops in the database is suitable"
st.subheader('Crop Recommended')
st.subheader(str(crop_name))
