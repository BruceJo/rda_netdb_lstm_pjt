import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM
from math import sqrt
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from keras.models import model_from_json

from tensorflow import keras
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler

envir = pd.read_csv('C:/nong/환경데이터.csv',header=0)


growth = pd.read_csv('C:/nong/생장데이터.csv',header=0)

scale_cols = ['온도 (℃)','습도 (%)']

env_feature_cnt = len(scale_cols) # 환경 변수 개수

growth_feature_cnt = 1 # 생육데이터는 생장길이 하나만을 이용하니까.

sample1_growth = growth[["DeltaS3"]].iloc[0:,:]
y_values = sample1_growth.values
print(type(y_values))
print(y_values.shape)
x_values = envir[scale_cols].values

x_train_size = int(len(x_values)*0.8)
y_train_size = int(len(y_values)*0.8)
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_values)
x_scaled.reshape(360,7,env_feature_cnt)

y_scaled = y_values
train_x = x_scaled[:x_train_size,:]
test_x = x_scaled[x_train_size:,:]
train_y = y_scaled[:y_train_size,:]
test_y = y_scaled[y_train_size:,:]

train_reshape1 = x_train_size//7
test_reshape1 = (len(x_values) - train_reshape1*7)//7

train_x = train_x.reshape((train_reshape1,7,env_feature_cnt))
test_x = test_x.reshape((test_reshape1,7,env_feature_cnt))

test_y_1 = test_y

train_x = train_x.reshape((train_x.shape[0], train_x.shape[1],env_feature_cnt))
test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], env_feature_cnt))
train_y = train_y.reshape((train_y.shape[0], 1,growth_feature_cnt))
test_y = test_y.reshape((test_y.shape[0], 1,growth_feature_cnt))

model = Sequential()
model.add(LSTM(50,input_shape=(train_x.shape[1],train_x.shape[2]),activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(train_x[:len(train_y)],train_y,epochs=500,batch_size=150,verbose=2,shuffle=False)

# pyplot.plot(history.history['loss'],label='train')
# pyplot.legend()
# pyplot.show()

# split into 3 sets of new data
test_X_1, test_X_2, test_Y_1, test_Y_2 = train_test_split(test_x[:len(test_y)], test_y, test_size=0.50, random_state=1)
test_X_2, test_X_3, test_Y_2, test_Y_3 = train_test_split(test_X_2, test_Y_2, test_size=0.50, random_state=1)

predictions = model.predict(test_X_1)

old_loss = model.evaluate(test_X_1,test_Y_1)

model_json = model.to_json()
with open("old_model.json","w") as json_file:
    json_file.write(model_json)

with open("new_model.json","w") as json_file:
    json_file.write(model_json)

model.save_weights("old_model.h5")
model.save_weights("new_model.h5")

json_file = open('new_model.json','r')
loaded_model_json = json_file.read()

json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("new_model.h5")

model = loaded_model

opt = SGD(learning_rate=0.001, momentum=0.9)
model.compile(optimizer=opt,loss='binary_crossentropy')

model.fit(test_X_2, test_Y_2, epochs=500, batch_size=150, verbose=2, shuffle=False)

predictions = model.predict(test_X_3)

for pre_val,act_val in zip(predictions,test_Y_3):
    print("Predicted:", pre_val)
    print("Actual:", act_val)

new_loss = model.evaluate(test_X_3,test_Y_3)

if new_loss < old_loss:
    # serialize old_model and new_model to JSON
    model_json = model.to_json()
    with open("old_model.json", "w") as json_file:
        json_file.write(model_json)
    with open("new_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("old_model.h5")
    model.save_weights("new_model.h5")
else:
    # serialize old_model and new_model to JSON
    model_json = model.to_json()
    with open("new_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("new_model.h5")

# mogodb에 mse저장
# from pymongo import MongoClient
# def get_database():
#     CONNECTION_STRING = "mongodb://netdb:netdb3230!@203.255.77.192:27017/"

#     client = MongoClient(CONNECTION_STRING)
#     return client["TestAPI"]

# if __name__ == "__main__":
#     dbname = get_database()
#     print(dbname)

# collection_name = dbname["MSE"]
# new_mse = collection_name.find_one({"_id":0})
# new_mse["LSTM"] = new_loss

# collection_name.replace_one({"_id":0},new_mse)