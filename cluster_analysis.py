import argparse
import pickle
import csv
from sklearn.cluster import SpectralClustering
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment

# Prepare parser

def args_parser_mask():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--algorithm', type=str, default='sub_fedavg', help='model name')
    parser.add_argument('--model', type=str, default='lenet5', help='model name')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        help="name of dataset: mnist, cifar10, cifar100")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--n_conv_layer', type=int, default=2, help='number of conv layers')
    parser.add_argument('--percentage', type=float, default=0.2, help='percentage of selected clients')
    parser.add_argument('--top', type=str, help='whether we select top overlapped networks or worst')
    parser.add_argument('--layers_from_last', type=int, default=5,help='whether we select top overlapped networks or worst')
    parser.add_argument('--n_cluster', type=int, default=5,help='whether we select top overlapped networks or worst')

    args = parser.parse_args()
    return args

def calculate_label_affinity_list(labels,label_list):
    affinity_list = []
    n_client = len(label_list)

    for idx in range(n_client):
        label_similarity = len(set(labels)&set(label_list[idx]))
        affinity_list.append(label_similarity)
    
    return affinity_list

def calculate_affinity_based_on_network(binary_mask_target,binary_mask_list_all,n_conv_layer=0,layer_from_last=5):
    affinity_list = []
    n_client = len(binary_mask_list_all)
    n_layer = len(binary_mask_target)

    for c_idx in range(n_client):
        total_num = 0
        overlap = 0

        for l in range(n_layer):
            if n_layer - layer_from_last > l:
                continue
            if l >= n_conv_layer:
                input_size = len(binary_mask_target[l])
                output_size = len(binary_mask_target[l][0])

                for i in range(input_size):
                    for o in range(output_size):
                        if binary_mask_target[l][i][o] == 1:
                            total_num += 1
                            if binary_mask_list_all[c_idx][l][i][o] == 1:
                                overlap += 1
            else:
                # conv layer
                n_dim_0 = len(binary_mask_target[l])
                n_dim_1 = len(binary_mask_target[l][0])
                n_dim_2 = len(binary_mask_target[l][0][0])
                n_dim_3 = len(binary_mask_target[l][0][0][0])
                for i in range(n_dim_0):
                    for j in range(n_dim_1):
                        for k in range(n_dim_2):
                            for m in range(n_dim_3):
                                if binary_mask_target[l][i][j][k][m] == 1:
                                    total_num += 1
                                    if binary_mask_list_all[c_idx][l][i][j][k][m] == 1:
                                        overlap += 1

        affinity_list.append(overlap/total_num)


    return affinity_list

def calculate_affinity_based_on_weight_divergence_of_locally_trained_models(locally_trained_weights_list,target_idx,n_conv_layer):
    n_client = len(locally_trained_weights_list)
    n_layer = int(len(locally_trained_weights_list[0])/2)
    affinity_based_on_weight_divergence_list = []

    weights_target = locally_trained_weights_list[target_idx]

    for c_idx in range(n_client):
        weight_divergence = 0 

        for l in range(n_layer):
            if l >= n_conv_layer:
                input_size = len(weights_target[l*2])
                output_size = len(weights_target[l*2][0])

                for o in range(output_size):
                    for i in range(input_size):
                    
                        weight_divergence += (weights_target[l*2][i][o] - locally_trained_weights_list[c_idx][l*2][i][o])**2
                        
                # Bias
                bias_input_size = len(weights_target[l*2+1])
                for o in range(bias_input_size):
                    weight_divergence += (weights_target[l*2+1][o] - locally_trained_weights_list[c_idx][l*2+1][o])**2
            else:
                # conv layer
                n_dim_0 = len(weights_target[l*2])
                n_dim_1 = len(weights_target[l*2][0])
                n_dim_2 = len(weights_target[l*2][0][0])
                n_dim_3 = len(weights_target[l*2][0][0][0])
                for i in range(n_dim_0):
                    for j in range(n_dim_1):
                        for k in range(n_dim_2):
                            for m in range(n_dim_3):
                                

                                weight_divergence += (weights_target[l*2][i][j][k][m] - locally_trained_weights_list[c_idx][l*2][i][j][k][m])**2

                for o in range(n_dim_3):
                
                    weight_divergence += (weights_target[l*2+1][o] - locally_trained_weights_list[c_idx][l*2+1][o])**2


        affinity_based_on_weight_divergence_list.append(weight_divergence)

    return affinity_based_on_weight_divergence_list


args = args_parser_mask()

# Import masks

mask_list = []
label_list = []
final_weights_list = []
for idx in range(args.num_users):
    file_name_mask = "src/data/masks/" + str(args.algorithm) +"_"+str(args.model) + "_" + str(args.num_users)  + "_" + str(args.dataset) + "_" + str(args.seed) + "_client_id_" + str(idx) + ".pickle"
    # open a file, where you stored the pickled data
    file = open(file_name_mask, 'rb')

    # dump information to that file
    mask = pickle.load(file)

    mask_list.append(mask)


    # Import labels
    file_name_weights = "src/data/labels/" + str(args.algorithm) +"_"+str(args.model)+ "_" + str(args.num_users) + "_" + str(args.dataset) + "_" + str(args.seed) + "_client_id_" + str(idx) + ".pickle"
    
    file = open(file_name_weights, 'rb')
    label = pickle.load(file)

    label_list.append(label)

    file_name_weights = "src/data/weights/" + str(args.algorithm) +"_"+str(args.model)+ "_" + str(args.num_users) + "_" + str(args.dataset) + "_" + str(args.seed) + "_client_id_" + str(idx) + ".pickle"
    file = open(file_name_weights, 'rb')

    # dump information to that file
    weights = pickle.load(file)

    weights_list = []
    for tensor in weights.items():
        
        weights_list.append(tensor[1])

    
    final_weights_list.append(weights_list)


# clustering based on groung truth labels

# create affinity matrix

label_affinity_mat = []
for idx in range(args.num_users):
    label_affinity_mat.append([])

    labels = label_list[idx]

    for ref_idx in range(args.num_users):
        ref_labels =label_list[ref_idx]

        affinity = float(len(set(labels)&set(ref_labels)))
        label_affinity_mat[idx].append(affinity)


clustering_base = SpectralClustering(n_clusters=args.n_cluster,
        affinity="precomputed",
        random_state=0)

clustering_base.fit_predict(label_affinity_mat)        

print(clustering_base.labels_)

for idx in range(args.num_users):
    print(label_list[idx])

for idx in range(args.num_users):
    label_affinity_mat[idx][idx] = 0

silhouette_score_base = sklearn.metrics.silhouette_score(label_affinity_mat,clustering_base.labels_,metric="precomputed")

print("silhouette_score = ",silhouette_score_base)



# clustering based on trained model

## clustering based on mask overlap ratio

mask_overlap_affinity_mat = []

for idx in range(args.num_users):
    affinity_list = calculate_affinity_based_on_network(mask_list[idx],mask_list,n_conv_layer=2,layer_from_last=5)
    mask_overlap_affinity_mat.append(affinity_list)



clustering_mask_overlap = SpectralClustering(n_clusters=args.n_cluster,
        affinity="precomputed",
        random_state=0)

clustering_mask_overlap.fit_predict(mask_overlap_affinity_mat)        

print(clustering_mask_overlap.labels_)


## density within cluster analysis 

for idx in range(args.num_users):
    mask_overlap_affinity_mat[idx][idx] = 0

silhouette_score_mask = sklearn.metrics.silhouette_score(mask_overlap_affinity_mat,clustering_mask_overlap.labels_,metric="precomputed")

print("silhouette_score = ",silhouette_score_mask)


## Classification accuracy

cm = confusion_matrix(clustering_base.labels_, clustering_mask_overlap.labels_)

def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)

indexes = linear_assignment(_make_cost_m(cm))
js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
cm2 = cm[:, js]

accuracy_mask = np.trace(cm2) / np.sum(cm2)

print("clustering accuracy based on mask overlap = ",accuracy_mask)


## clustering based on network similarity


weight_divergence_affinity_mat = []

for idx in range(args.num_users):
    affinity_list = calculate_affinity_based_on_weight_divergence_of_locally_trained_models(final_weights_list,idx,n_conv_layer=2)
    weight_divergence_affinity_mat.append(affinity_list)

## normalize matrix

max_each_row_list = [max(p) for p in weight_divergence_affinity_mat]
max_value = max(max_each_row_list)

for idx in range(args.num_users):
    for ref_idx in range(args.num_users):
        weight_divergence_affinity_mat[idx][ref_idx] = 1 - weight_divergence_affinity_mat[idx][ref_idx]/max_value

## 1 - distance matrix

clustering_weight_div = SpectralClustering(n_clusters=args.n_cluster,
        affinity="precomputed",
        random_state=0)

clustering_weight_div.fit_predict(weight_divergence_affinity_mat)        

print(clustering_weight_div.labels_)

## density within cluster analysis 

for idx in range(args.num_users):
    weight_divergence_affinity_mat[idx][idx] = 0

silhouette_score_weight_div = sklearn.metrics.silhouette_score(weight_divergence_affinity_mat,clustering_weight_div.labels_,metric="precomputed")

print("silhouette_score = ",silhouette_score_weight_div)



## Classification accuracy

cm = confusion_matrix(clustering_base.labels_, clustering_weight_div.labels_)

indexes = linear_assignment(_make_cost_m(cm))
js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
cm2 = cm[:, js]

accuracy_weight_div = np.trace(cm2) / np.sum(cm2)

print("clustering accuracy based on mask overlap = ",accuracy_weight_div)


# save results

csv_rows_each_round = [[args.num_users,args.algorithm,args.model,args.dataset,args.seed,args.n_cluster,silhouette_score_mask,accuracy_mask,silhouette_score_weight_div,accuracy_weight_div,silhouette_score_base]]
with open('src/data/log/cluster_analysis.csv', 'a') as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(csv_rows_each_round)
