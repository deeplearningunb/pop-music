import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from minisom import MiniSom
import pickle
import matplotlib.pyplot as plt
import os

model_filename = "som.pickle"

print("Loading data...")
dataset = pd.read_csv("track_metadata.csv")

print(f"lines: {dataset.shape[0]}, columns: {dataset.shape[1]}")

# there are too many ids and titles to OneHotEncode
ts = [("text", OneHotEncoder(sparse=False), list(range(0, 5))),("minmax", MinMaxScaler(), list(range(6, 12)))]
transformer = ColumnTransformer(transformers=ts)
print("Transforming...")
t_dataset = transformer.fit_transform(dataset)

if (not os.path.exists(model_filename)):
    print(f"Could not find {model_filename}, will train")

    # minisom recommends at least 4*sqrt(N) neurons
    # 515576 samples require 3651 neurons
    # a square grid of 60 x 60 should be enough
    samples = t_dataset.shape[0]
    rec_xydim = math.ceil(math.sqrt(5*math.sqrt(samples)))
    learn_rate = 0.25
    iterations = 250
    som = MiniSom(x=rec_xydim, y=rec_xydim, input_len=t_dataset.shape[1], learning_rate=learn_rate, random_seed=0)
    print(f"Train square grid with {rec_xydim} neurons on each side, learning rate of {learn_rate}")
    som.train(t_dataset, iterations, verbose=True)

    print(f"Saving SOM as {model_filename}")
    outfile = open(model_filename, "wb")
    pickle.dump(som, outfile)
    outfile.close()

print(f"Restoring SOM from {model_filename}")
infile = open(model_filename, "rb")
som = pickle.load(infile)
infile.close()

print("Plotting...")
distance_map = som.distance_map()
#print(distance_map)

plt.pcolormesh(distance_map)
plt.colorbar()
plt.show()
