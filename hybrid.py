import numpy as np
import matplotlib.pyplot as plt
from pylab import bone, pcolor, colorbar, plot, show
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from MiniSom.minisom import MiniSom
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# ----------------------- dataset import ------------------------------

BASE = os.getcwd()
DATASET_DIR = BASE + '/Dataset/'
FILE_NAME = DATASET_DIR + 'Credit_Card_Applications.csv'
MODEL_DIR = BASE + '/Models/myModel'

# ----------------------------------------------------------------------

# getting the features
dataset = pd.read_csv(FILE_NAME)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

# Training the SOM
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X, num_iteration=100)

# visualizing the results
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]], )

show()

# finding the frauds
mapping = som.win_map(X)
frauds = np.concatenate((mapping[(3, 1)], mapping[(9, 2)]), axis=0)
frauds = sc.inverse_transform(frauds)

# creating the matix of features
customers = dataset.iloc[:, 1:].values

# creating the dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

sc = StandardScaler()
customers = sc.fit_transform(customers)
# defining the ANN
classifier = Sequential()
classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu', input_dim=15))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(customers, is_fraud, batch_size=1, epochs=2)
classifier.save(MODEL_DIR, overwrite=True)

# prediction
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]
