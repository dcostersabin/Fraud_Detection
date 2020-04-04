import numpy as np
import matplotlib.pyplot as plt
from pylab import bone, pcolor, colorbar, plot, show
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from MiniSom.minisom import MiniSom

# ----------------------- dataset import ------------------------------

BASE = os.getcwd()
DATASET_DIR = BASE + '/Dataset/'
FILE_NAME = DATASET_DIR + 'Credit_Card_Applications.csv'

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
frauds = np.concatenate((mapping[(8, 2)], mapping[(6, 2)]), axis=0)
frauds = sc.inverse_transform(frauds)
