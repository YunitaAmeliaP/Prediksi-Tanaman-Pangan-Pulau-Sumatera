import streamlit as st

st.title("Prediksi Hasil Panen Tanaman Pangan di Pulau Sumatera")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('/content/Data_tanaman_pangan_Sumatera.csv')

df

# Exploration Data

df.isnull().sum()

df.describe()

sns.lineplot(x="Tahun", y="Produksi", data=df)

plt.figure(figsize=(25,10))
sns.barplot(x="Provinsi", y="Produksi", data=df)
plt.xticks(rotation=90)

sns.jointplot(x="Luas Panen", y="Produksi", data=df, kind='reg')

sns.barplot(x=df["Item"],y=df["Produksi"]) # Changed to use keyword arguments x and y

## 1. Jagung

jagung_df = df[df["Item"]=="Jagung"]
print(jagung_df.shape)

plt.figure(figsize=(25,10))
sns.barplot(x="Tahun", y="Produksi", data=jagung_df) # Use keyword arguments for x and y
plt.xticks(rotation=45)

# plt.figure(figsize=(13,10))
# sns.barplot("Provinsi","Produksi",data=jagung_df) # Original line causing error
plt.figure(figsize=(13,10))
sns.barplot(x="Provinsi", y="Produksi", data=jagung_df)

sns.jointplot(x="Luas Panen", y="Produksi", data=jagung_df, kind="reg")

sns.jointplot(x="Curah hujan", y="Produksi", data=jagung_df, kind="reg")

sns.jointplot(x="Suhu rata-rata", y="Produksi", data=jagung_df, kind="reg")

## 2. Ubi Kayu

ubik_df = df[df["Item"]=="Ubi Kayu"]
print(ubik_df.shape)

plt.figure(figsize=(25,10))
sns.barplot(x="Tahun", y="Produksi", data=ubik_df) # Use keyword arguments for x and y
plt.xticks(rotation=45)

plt.figure(figsize=(13,10))
# Explicitly specify x and y using keyword arguments
sns.barplot(x="Provinsi", y="Produksi", data=ubik_df)

sns.jointplot(x="Luas Panen", y="Produksi", data=ubik_df, kind="reg")

sns.jointplot(x="Curah hujan", y="Produksi", data=ubik_df, kind="reg")

## 3. Ubi Jalar

ubij_df = df[df["Item"]=="Ubi Jalar"]
print(ubij_df.shape)

# %%
# Pastikan ubij_df sudah terdefinisi sebelum menjalankan baris ini
plt.figure(figsize=(25,10))
# Explicitly specify x and y using keyword arguments
sns.barplot(x="Tahun", y="Produksi", data=ubij_df) # Garis ini memerlukan ubij_df
plt.xticks(rotation=45)

# Use keyword arguments for x and y to avoid ambiguity
sns.jointplot(x="Luas Panen", y="Produksi", data=ubij_df, kind="reg")

## 4. Kacang Tanah

kac_df = df[df["Item"]=="Kacang Tanah"]
print(kac_df.shape)

plt.figure(figsize=(25,10))
# Explicitly specify x and y using keyword arguments
sns.barplot(x="Tahun", y="Produksi", data=kac_df)
plt.xticks(rotation=45)

plt.figure(figsize=(13,10))
# Explicitly specify x and y using keyword arguments
sns.barplot(x="Provinsi", y="Produksi", data=kac_df)

sns.jointplot(x="Luas Panen", y="Produksi", data=kac_df, kind="reg")

## 5. Kedelai

ked_df = df[df["Item"]=="Kedelai"]
print(ked_df.shape)

plt.figure(figsize=(25,10))
sns.barplot(x="Tahun", y="Produksi", data=ked_df)
plt.xticks(rotation=45)

plt.figure(figsize=(13,10))
# Explicitly specify x and y using keyword arguments
sns.barplot(x="Provinsi", y="Produksi", data=ked_df)

sns.jointplot(x="Luas Panen", y="Produksi", data=ked_df, kind="reg")

# Select only numeric columns before calculating correlation
numeric_df = df.select_dtypes(include=np.number)
C_mat = numeric_df.corr()

fig = plt.figure(figsize = (8,8))
sns.heatmap(C_mat, square = True, annot=True)
plt.show()

# Preprocessing Data

df_one = pd.get_dummies(df, columns=['Provinsi', 'Item'], prefix = ['Provinsi', 'Item'])

df_one

df_one = df_one.drop(columns = 'Tahun')

df_one.head()

df_one.info()

# Preprocessing Data

X_reg=df_one.loc[:, df_one.columns != 'Produksi'].values
y_reg=df_one['Produksi'].values

X_reg

y_reg = y_reg.reshape(-1,1)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_reg,y_reg,test_size=0.2, random_state=42)
print("X_train :",X_train.shape)
print("X_test :",X_test.shape)
print("y_train :",y_train.shape)
print("y_test :",y_test.shape)

## Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, :] = sc.fit_transform(X_train[:, :])
X_test[:, :] = sc.transform(X_test[:, :])
y_train[:, :] = sc.fit_transform(y_train[:, :])
y_test[:, :] = sc.transform(y_test[:, :])

# Model Selection

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score

## 1. Random Forest

regresi = RandomForestRegressor(max_depth=4, random_state=0, n_estimators=1000)
regresi.fit(X_train, np.ravel(y_train))
y_pred = regresi.predict(X_test)

mse_rf = mse(y_pred, y_test)
print('MSE =', mse_rf)
print('MAE =', mae(y_pred, y_test))
print('RMSE =', np.sqrt(mse_rf))
print('R2 Score =', r2_score(y_pred, y_test))

y_pred = regresi.predict(X_test)

# Reshape y_pred to be a 2D array with one column
y_pred = y_pred.reshape(-1, 1)

y_pred = sc.inverse_transform(y_pred)

mse_rf = mse(y_pred, y_test)
print('MSE =', mse_rf)
print('MAE =', mae(y_pred, y_test))
print('RMSE =', np.sqrt(mse_rf))
print('R2 Score =', r2_score(y_pred, y_test))

## 2. Decision Tree

dtr = DecisionTreeRegressor(max_depth=4)
dtr.fit(X_train,np.ravel(y_train))
dtr_pred = dtr.predict(X_test)

mse_dtr = mse(dtr_pred, y_test)
print('MSE =', mse_dtr )
print('MAE =', mae(dtr_pred, y_test))
print('RMSE =', np.sqrt(mse_dtr))
print('R2 Score =', r2_score(dtr_pred, y_test))

# Reshape dtr_pred to be a 2D array with one column
dtr_pred = dtr_pred.reshape(-1, 1)

dtr_pred = sc.inverse_transform(dtr_pred)

mse_dtr = mse(dtr_pred, y_test)
print('MSE =', mse_dtr )
print('MAE =', mae(dtr_pred, y_test))
print('RMSE =', np.sqrt(mse_dtr))
print('R2 Score =', r2_score(dtr_pred, y_test))

## 3. Gradient Boosting Regressor

gbr = GradientBoostingRegressor(n_estimators=1000,max_depth=4,random_state=0)
gbr.fit(X_train, np.ravel(y_train))
gbr_pred = gbr.predict(X_test)

mse_gbr = mse(gbr_pred, y_test)
print('MSE =', mse_gbr)
print('MAE =', mae(gbr_pred, y_test))
print('RMSE =', np.sqrt(mse_gbr))
print('R2 Score =', r2_score(gbr_pred, y_test))

# Reshape gbr_pred to be a 2D array with one column
gbr_pred = gbr_pred.reshape(-1, 1)

# Now, inverse_transform should work correctly
gbr_pred = sc.inverse_transform(gbr_pred)

mse_gbr = mse(gbr_pred, y_test)
print('MSE =', mse_gbr)
print('MAE =', mae(gbr_pred, y_test))
print('RMSE =', np.sqrt(mse_gbr))
print('R2 Score =', r2_score(gbr_pred, y_test))

## 4. Extra Tree

xtr = ExtraTreesRegressor(n_estimators=100, random_state=0)
xtr.fit(X_train, np.ravel(y_train))
xtr_pred = xtr_pred = xtr.predict(X_test)

mse_xtr = mse(xtr_pred, y_test)
print('MSE =', mse_xtr)
print('MAE =', mae(xtr_pred, y_test))
print('RMSE =', np.sqrt(mse_xtr))
print('R2 Score =', r2_score(xtr_pred, y_test))

## 5. SVR

from sklearn.svm import SVR
regresiSVR =SVR(kernel='rbf')
regresiSVR.fit(X_train,np.ravel(y_train))
pred=regresiSVR.predict(X_test)

mse_svr = mse(pred, y_test)
print('MSE =', mse_svr)
print('MAE =', mae(pred, y_test))
print('RMSE =', np.sqrt(mse_svr))
#print('R2 Score =', r2_score(pred, y_test))
print(regresiSVR.score(X_test,y_test))

# Reshape pred to be a 2D array with one column
pred = pred.reshape(-1, 1)

# Now, inverse_transform should work correctly
pred = sc.inverse_transform(pred)

mse_svr = mse(pred, y_test)
print('MSE =', mse_svr)
print('MAE =', mae(pred, y_test))
print('RMSE =', np.sqrt(mse_svr))
print('R2 Score =', r2_score(pred, y_test))
#print(regresiSVR.score(X_test,y_test))

## 6. XGBOOST

from xgboost import XGBRegressor

XGBModel = XGBRegressor()
XGBModel.fit(X_train,y_train , verbose=True)
xgb_pred = XGBModel.predict(X_test)

mse_xgb = mse(xgb_pred, y_test)
print('MSE =', mse_xgb)
print('MAE =', mae(xgb_pred, y_test))
print('RMSE =', np.sqrt(mse_xgb))
print('R2 Score =', r2_score(xgb_pred, y_test))

# Reshape xgb_pred to be a 2D array with one column
xgb_pred = xgb_pred.reshape(-1, 1)

# Now, inverse_transform should work correctly
xgb_pred = sc.inverse_transform(xgb_pred)

mse_xgb = mse(xgb_pred, y_test)
print('MSE =', mse_xgb)
print('MAE =', mae(xgb_pred, y_test))
print('RMSE =', np.sqrt(mse_xgb))
print('R2 Score =', r2_score(xgb_pred, y_test))

## 7.ANN

import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()

model.add(Dense(60, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))
model.add(Dense(30, activation='relu',kernel_initializer='normal'))

model.add(Dense(1,activation='linear',kernel_initializer='normal'))

model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])
model.summary()

from keras.callbacks import History
history = History()

# Ensure X_train and y_train are float32
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')

History = model.fit(X_train,y_train,batch_size = 32, epochs= 500,verbose=1,validation_split=0.2,callbacks=[history])
#loss_df=pd.DataFrame(model.history.history)

plt.figure(figsize=(14,8))
plt.plot(History.history['mean_squared_error'])
plt.plot(History.history['val_mean_squared_error'])
plt.title('Mean squared Error', fontsize=14)
plt.ylabel('mean_squared_error', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend(['train', 'test'], loc='upper left')
plt.show()
 # summarize history for loss
plt.figure(figsize=(14,8))
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss', fontsize=14)
plt.ylabel('Loss',fontsize=12)
plt.xlabel('Epoch',fontsize=12)
plt.legend(['train', 'test'], loc='upper left')
plt.show()

loss_df=pd.DataFrame(model.history.history)
loss_df.plot(figsize=(14,8))
plt.title('mean_squared_error')
plt.ylabel('mean_squared_error')
plt.xlabel('epoch')

# Ensure X_test is also of type float32 before prediction
X_test = X_test.astype('float32')

y_prediksi = model.predict(X_test)

mse2 = mse(y_prediksi, y_test)
print('MSE =', mse2)
print('MAE =', mae(y_prediksi, y_test))
print('RMSE =', np.sqrt(mse2))
print('R2 Score =', r2_score(y_prediksi, y_test))

y_prediksi = sc.inverse_transform(y_prediksi)

mse2 = mse(y_prediksi, y_test)
print('MSE =', mse2)
print('MAE =', mae(y_prediksi, y_test))
print('RMSE =', np.sqrt(mse2))
print('R2 Score =', r2_score(y_prediksi, y_test))

plt.scatter(y_test, y_prediksi)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("ANN Model")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # Import numpy if not already imported

# Assuming y_test and y_prediksi are already defined from previous steps

sns.regplot(x=y_test, y=y_prediksi)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("ANN Model")
plt.show()

# Evualating Model

# XTree best from other model, so


###predicting the test set results
print(np.concatenate((xtr_pred.reshape(len(xtr_pred),1), y_test.reshape(len(y_test),1)),1))


xtr_pred = xtr_pred.reshape(-1, 1)
xtr_pred = sc.inverse_transform(xtr_pred)
y_test = sc.inverse_transform(y_test)

print(np.concatenate((xtr_pred.reshape(len(xtr_pred),1), y_test.reshape(len(y_test),1)),1))

mse_xtr = mse(xtr_pred, y_test)
print('MSE =', mse_xtr)
print('MAE =', mae(xtr_pred, y_test))
print('RMSE =', np.sqrt(mse_xtr))
print('R2 Score =', r2_score(xtr_pred, y_test))

plt.scatter(y_test, xtr_pred)
plt.xlabel("Actual Values",fontsize=12)
plt.ylabel("Predicted Values",fontsize=12)
plt.title("Extra Tree Model",fontsize=14)
plt.show()

sns.regplot(x=y_test, y=xtr_pred)
plt.xlabel("Actual Values", fontsize=12)
plt.ylabel("Predicted Values", fontsize=12)
plt.title("Extra Tree Model", fontsize=14)
plt.show()

sns.regplot(x=y_test, y=xtr_pred)
sns.regplot(x=y_test, y=y_prediksi)
plt.xlabel("Actual Values", fontsize=12)
plt.ylabel("Predicted Values", fontsize=12)
plt.title("Extra Tree Model", fontsize=14)
plt.legend(['Extra Tree', 'ANN'], loc='upper left')
plt.show()

# Fix the regplot calls by explicitly using x= and y=
sns.regplot(x=y_test, y=xtr_pred, marker="o")
sns.regplot(x=y_test, y=y_prediksi, marker="x")
plt.xlabel("Actual Values", fontsize=12)
plt.ylabel("Predicted Values", fontsize=12)
plt.title("Extra Tree Model", fontsize=14)
plt.legend(['Extra Tree', 'ANN'], loc='upper left')
plt.show()

# Fix the sns.barplot call by explicitly using x= and y=
plt.figure(figsize=(10,5))
sns.barplot(x=names, y=scorre)
#plt.bar(names,scorre, color='green')
plt.xlabel('Model', fontsize=12)
plt.ylabel("R^2 Score", fontsize=12)
plt.ylim(0,1)
plt.title('Perbandingan Model',fontsize=14)
plt.show()

test_1 = np.array([[1401, 86.53, 27.03, 319507, 0,0,0,0,0,0,0,1,1,0,0,0,0]])

test_1_pred= model.predict(test_1)

test_1_pred

test_2_pred= xtr.predict(test_1)

test_2_pred

test_2_pred

test_2_pred = xtr.predict(test_1)

# Reshape the 1D array to a 2D array with one column
test_2_pred = test_2_pred.reshape(-1, 1)

test_2_pred = sc.inverse_transform(test_2_pred)

test_2_pred
