import pandas as pd
import numpy as np

dataset = "cifar10"
anneal_factor = 0.5
training_rounds = 100
cosine = True
transfer = "positive"
sparse_value = 0.2
algorithm = "create_data_for_mask_positive_transfer_correlation"

file_name_common = "src/data/positive_transfer/dataset_" + str(dataset) + "training_round_" + str(training_rounds) + "anneal_factor_" +str(anneal_factor) + "similar_clients_" + str(transfer) + "sparcities_" + str(sparse_value) +"algorithm_" + str(algorithm)

sim_file_name = file_name_common + "mask_distance.csv"
pt_file_name = file_name_common + "accuracy_list.csv"

X = pd.read_csv(sim_file_name,header=None)
y = pd.read_csv(pt_file_name,header=None)
y = y.transpose()




feature_idx = 0

import scipy.stats as stats
r = stats.pearsonr(X[feature_idx], y)
print("pearson correlation= ",r)

import matplotlib.pyplot as plt
plt.scatter(X[feature_idx], y, alpha=0.5)
plt.show()

# t test

threshold = 0.18

high_mask_distance_list = []
low_mask_distance_list = []

n_row = len(y)

for i in range(n_row):
    if X[feature_idx][i] >= threshold:
        high_mask_distance_list.append(y.iloc[i].item())
    else:
        low_mask_distance_list.append(y.iloc[i].item())

print(stats.ttest_ind(low_mask_distance_list, high_mask_distance_list, trim=.2))

mymodel = np.poly1d(np.polyfit(X[feature_idx].values.tolist(), np.squeeze(y.values), 1))

myline = np.linspace(0.14, 0.22, 100)

plt.scatter(X[feature_idx], y)
plt.plot(myline, mymodel(myline))
plt.xlabel("Mask Hamming Distance")
plt.ylabel("Best Test Accuracy")
plt.show()

from sklearn.metrics import r2_score
print(r2_score(y, mymodel(X[feature_idx])))