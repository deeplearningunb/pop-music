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
hott_marker_x_filename = "hott_marker_x.pickle"
hott_marker_y_filename = "hott_marker_y.pickle"
fam_marker_x_filename = "fam_marker_x.pickle"
fam_marker_y_filename = "fam_marker_y.pickle"
wins_filename = "wins.pickle"

artist_hottnesss_limit = 0.5
artist_familiarity_limit = 0.5

print("Loading data...")
dataset = pd.read_csv("track_metadata.csv")

artist_hottnesss = dataset.loc[:, "artist_hotttnesss"].to_numpy()
artist_hott = [x > artist_hottnesss_limit for x in artist_hottnesss]

artist_familiarity = dataset.loc[:, "artist_familiarity"].to_numpy()
artist_familiar = [x > artist_familiarity_limit for x in artist_familiarity]

print("Original data")
for i, x in enumerate(dataset.columns):
    print(i, x)

print(f"lines: {dataset.shape[0]}, columns: {dataset.shape[1]}")

print("Will drop columns for training")
d_dataset = dataset.iloc[:, [0, 2, 4, 6, 9, 10, 11]]

print("Training data")
for i, x in enumerate(d_dataset.columns):
    print(i, x)

print(f"lines: {d_dataset.shape[0]}, columns: {d_dataset.shape[1]}")

t_dataset = d_dataset.to_numpy()

ts = [("text", OneHotEncoder(sparse=False), list(range(0, 3))),("minmax", MinMaxScaler(feature_range=(0, 1)), list(range(3, 7)))]
transformer = ColumnTransformer(transformers=ts)
print("Transforming...")
t_dataset = transformer.fit_transform(t_dataset)

if (not os.path.exists(model_filename)):
    print(f"Could not find {model_filename}, will train")

    # minisom recommends at least 4*sqrt(N) neurons
    # 515576 samples require 3651 neurons
    # a square grid of 60 x 60 should be enough
    samples = t_dataset.shape[0]
    rec_xydim = math.ceil(math.sqrt(5*math.sqrt(samples)))
    learn_rate = 0.5
    iterations = 1250
    n_distance = 1.0
    som = MiniSom(x=rec_xydim, y=rec_xydim, input_len=t_dataset.shape[1], sigma=n_distance, learning_rate=learn_rate, random_seed=0)
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
distance_map = som.distance_map().T
#print(distance_map)

print("Ploting Distance Map...")
plt.pcolormesh(distance_map)
plt.colorbar()
plt.title("Distance Map")
plt.savefig("distance_map.pdf")
plt.close()
print("Saved Distance Map")

if ((not os.path.exists(hott_marker_x_filename)) or (not os.path.exists(hott_marker_y_filename)) or (not os.path.exists(fam_marker_x_filename)) or (not os.path.exists(fam_marker_y_filename))):
    print("Could not find the persisted markers, will generate them")
    print("Generate markers for hottnesss and familiarity")
    hott_marker_x = []
    hott_marker_y = []
    fam_marker_x = []
    fam_marker_y = []
    for i, x in enumerate(t_dataset):
        if ((i % 100) == 0):
            print(f"{i}/{samples}")
        if (artist_hott[i]):
            hott_marker_x.append(som.winner(x)[0] + 0.5)
            hott_marker_y.append(som.winner(x)[1] + 0.5)
        if (artist_familiar[i]):
            fam_marker_x.append(som.winner(x)[0] + 0.5)
            fam_marker_y.append(som.winner(x)[1] + 0.5)
    # save them
    print(f"Saving {hott_marker_x_filename}")
    outfile = open(hott_marker_x_filename, "wb")
    pickle.dump(hott_marker_x, outfile)
    outfile.close()

    print(f"Saving {hott_marker_y_filename}")
    outfile = open(hott_marker_y_filename, "wb")
    pickle.dump(hott_marker_y, outfile)
    outfile.close()

    print(f"Saving {fam_marker_x_filename}")
    outfile = open(fam_marker_x_filename, "wb")
    pickle.dump(fam_marker_x, outfile)
    outfile.close()

    print(f"Saving {fam_marker_y_filename}")
    outfile = open(fam_marker_y_filename, "wb")
    pickle.dump(fam_marker_y, outfile)
    outfile.close()

print(f"Restoring {hott_marker_x_filename}")
infile = open(hott_marker_x_filename, "rb")
hott_marker_x = pickle.load(infile)
infile.close()

print(f"Restoring {hott_marker_y_filename}")
infile = open(hott_marker_y_filename, "rb")
hott_marker_y = pickle.load(infile)
infile.close()

print(f"Restoring {fam_marker_x_filename}")
infile = open(fam_marker_x_filename, "rb")
fam_marker_x = pickle.load(infile)
infile.close()

print(f"Restoring {fam_marker_y_filename}")
infile = open(fam_marker_y_filename, "rb")
fam_marker_y = pickle.load(infile)
infile.close()

print("Ploting Familar Hott Map")
plt.pcolormesh(distance_map)
plt.colorbar()
plt.title("Distribution of artists' Hottnesss and Familiarity")
plt.plot(hott_marker_x, hott_marker_y, linestyle="None", marker="o", mec="k", mfc="None", label="hott")
plt.plot(fam_marker_x, fam_marker_y, linestyle="None", marker="x", mec="r", mfc="None", label="familiar")
plt.legend()
plt.savefig("pop_fam.pdf")
plt.close()
print("Saved Familiar Hott Map")

print("\nAnalysis\n")
if (not os.path.exists(wins_filename)):
    print("Could not find map, generating it...")
    wins = som.win_map(t_dataset, return_indices=True)
    # save it
    print(f"Saving {wins_filename}")
    outfile = open(wins_filename, "wb")
    pickle.dump(wins, outfile)
    outfile.close()

print(f"Restoring {wins_filename}")
infile = open(wins_filename, "rb")
wins = pickle.load(infile)
infile.close()

print("\nTracks with artists that are familiar but not hott")
familiar_not_hott = [wins[(0, 16)], wins[(0, 18)], wins[(1, 13)]]
for fnt in familiar_not_hott:
        print(dataset.iloc[fnt, :].to_string())

print("\nTracks that are different but have familiar and hott artists")
diff_hott_fam = wins[(2, 15)]
print(dataset.iloc[diff_hott_fam, :].to_string())
