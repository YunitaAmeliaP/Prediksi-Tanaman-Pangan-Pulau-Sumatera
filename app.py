import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("Prediksi Hasil Panen Tanaman Pangan di Pulau Sumatera")

# Load data
df = pd.read_csv('Data_tanaman_pangan_Sumatera.csv')

st.write("## Data Preview")
st.dataframe(df)

# Data exploration
st.write("### Null Values per Column")
st.write(df.isnull().sum())

st.write("### Descriptive Statistics")
st.write(df.describe())

# Lineplot Produksi per Tahun
fig, ax = plt.subplots()
sns.lineplot(x="Tahun", y="Produksi", data=df, ax=ax)
st.pyplot(fig)

# Barplot Produksi per Provinsi
fig, ax = plt.subplots(figsize=(25,10))
sns.barplot(x="Provinsi", y="Produksi", data=df, ax=ax)
plt.xticks(rotation=90)
st.pyplot(fig)

# Jointplot Luas Panen vs Produksi
sns.jointplot(x="Luas Panen", y="Produksi", data=df, kind='reg')
st.pyplot(plt.gcf())
plt.clf()

# Barplot Item vs Produksi
fig, ax = plt.subplots()
sns.barplot(x="Item", y="Produksi", data=df, ax=ax)
st.pyplot(fig)

# Analisis per Item (Contoh: Jagung)
def plot_item(item_name):
    item_df = df[df["Item"] == item_name]
    st.write(f"### Data untuk {item_name}, jumlah data: {item_df.shape[0]}")
    
    fig, ax = plt.subplots(figsize=(25,10))
    sns.barplot(x="Tahun", y="Produksi", data=item_df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(13,10))
    sns.barplot(x="Provinsi", y="Produksi", data=item_df, ax=ax)
    st.pyplot(fig)
    
    for feature in ["Luas Panen", "Curah hujan", "Suhu rata-rata"]:
        if feature in item_df.columns:
            sns.jointplot(x=feature, y="Produksi", data=item_df, kind="reg")
            st.pyplot(plt.gcf())
            plt.clf()

plot_item("Jagung")
plot_item("Ubi Kayu")
plot_item("Ubi Jalar")
plot_item("Kacang Tanah")
plot_item("Kedelai")

# Korelasi matriks (numeric only)
numeric_df = df.select_dtypes(include=np.number)
C_mat = numeric_df.corr()

fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(C_mat, square=True, annot=True, ax=ax)
st.pyplot(fig)

# Preprocessing
df_one = pd.get_dummies(df, columns=['Provinsi', 'Item'], prefix=['Provinsi', 'Item'])
df_one = df_one.drop(columns='Tahun')

st.write("## Data setelah one-hot encoding dan drop kolom Tahun")
st.dataframe(df_one.head())

# Split fitur dan target
X_reg = df_one.loc[:, df_one.columns != 'Produksi'].values
y_reg = df_one['Produksi'].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

st.write(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
st.write(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)

# Model dan evaluasi
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score

def evaluate_model(model, X_train, y_train, X_test, y_test, scaler_y, model_name):
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test).reshape(-1,1)
    
    # Inverse transform y_pred and y_test
    y_pred_inv = scaler_y.inverse_transform(y_pred)
    y_test_inv = scaler_y.inverse_transform(y_test)
    
    mse_val = mse(y_test_inv, y_pred_inv)
    mae_val = mae(y_test_inv, y_pred_inv)
    rmse_val = np.sqrt(mse_val)
    r2_val = r2_score(y_test_inv, y_pred_inv)
    
    st.write(f"### {model_name} Evaluation")
    st.write(f"MSE = {mse_val}")
    st.write(f"MAE = {mae_val}")
    st.write(f"RMSE = {rmse_val}")
    st.write(f"R2 Score = {r2_val}")

# Random Forest
rf = RandomForestRegressor(max_depth=4, random_state=0, n_estimators=1000)
evaluate_model(rf, X_train, y_train, X_test, y_test, sc_y, "Random Forest Regressor")

# Decision Tree
dt = DecisionTreeRegressor(max_depth=4)
evaluate_model(dt, X_train, y_train, X_test, y_test, sc_y, "Decision Tree Regressor")

# Gradient Boosting
gbr = GradientBoostingRegressor(n_estimators=1000, max_depth=4, random_state=0)
evaluate_model(gbr, X_train, y_train, X_test, y_test, sc_y, "Gradient Boosting Regressor")

# Extra Trees
xtr = ExtraTreesRegressor(n_estimators=100, random_state=0)
evaluate_model(xtr, X_train, y_train, X_test, y_test, sc_y, "Extra Trees Regressor")

# SVR
svr = SVR(kernel='rbf')
evaluate_model(svr, X_train, y_train, X_test, y_test, sc_y, "Support Vector Regressor")

# XGBoost
xgb = XGBRegressor()
evaluate_model(xgb, X_train, y_train, X_test, y_test, sc_y, "XGBoost Regressor")

# ANN Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(60, kernel_initializer='normal', input_dim=X_train.shape[1], activation='relu'),
    Dense(30, activation='relu', kernel_initializer='normal'),
    Dense(1, activation='linear', kernel_initializer='normal')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
st.write("### ANN Model Summary")
model.summary(print_fn=st.write)

# Convert to float32
X_train_tf = X_train.astype('float32')
y_train_tf = y_train.astype('float32')

history = model.fit(X_train_tf, y_train_tf, batch_size=32, epochs=500, verbose=0, validation_split=0.2)

# Plot training history
fig, ax = plt.subplots(figsize=(14,8))
ax.plot(history.history['mean_squared_error'], label='Train MSE')
ax.plot(history.history['val_mean_squared_error'], label='Validation MSE')
ax.set_title('Mean Squared Error')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE')
ax.legend()
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(14,8))
ax.plot(history.history['loss'], label='Train Loss')
ax.plot(history.history['val_loss'], label='Validation Loss')
ax.set_title('Model Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
st.pyplot(fig)

# Predict ANN on test data
X_test_tf = X_test.astype('float32')
y_pred_ann = model.predict(X_test_tf)
y_pred_ann_inv = sc_y.inverse_transform(y_pred_ann)
y_test_inv = sc_y.inverse_transform(y_test)

mse_ann = mse(y_test_inv, y_pred_ann_inv)
mae_ann = mae(y_test_inv, y_pred_ann_inv)
rmse_ann = np.sqrt(mse_ann)
r2_ann = r2_score(y_test_inv, y_pred_ann_inv)

st.write("### ANN Model Evaluation")
st.write(f"MSE = {mse_ann}")
st.write(f"MAE = {mae_ann}")
st.write(f"RMSE = {rmse_ann}")
st.write(f"R2 Score = {r2_ann}")
