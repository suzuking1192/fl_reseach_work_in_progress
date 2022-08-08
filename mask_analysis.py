import argparse
import pickle
import csv

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

    args = parser.parse_args()
    return args

args = args_parser_mask()

# Import masks

mask_list = []
label_list = []
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



# Calculate mask difference 

n_client = len(mask_list)

top_label_similarity_list = []

select_users_num = int(n_client*args.percentage)

for c_idx in range(n_client):
    affinity_list = calculate_affinity_based_on_network(mask_list[c_idx],mask_list,args.n_conv_layer,args.layers_from_last)
    #print("affinity_list = ",affinity_list)
    label_1 = label_list[c_idx]
    print("label_base = ",label_1)
    
    # Get top 6 indicies based on affinity measures
    

    # if args.top == True:
    selected_idx = sorted(range(len(affinity_list)), key=lambda i: affinity_list[i])[-select_users_num:]
    # # else:

    # selected_idx = sorted(range(len(affinity_list)), key=lambda i: affinity_list[i])[:select_users_num]

    for ref_c_idx in selected_idx:
        if c_idx == ref_c_idx:
            pass
        else:
            label_2 = label_list[ref_c_idx]

            print("selected label = ",label_2)
            print("overlapped percentage = ", affinity_list[ref_c_idx])

            label_similarity = len(set(label_1)&set(label_2))

            top_label_similarity_list.append(label_similarity)

    
print("average label overlap in top 5 overlapped network",(sum(top_label_similarity_list)/len(top_label_similarity_list))/2)

csv_rows_each_round = [[args.num_users,args.algorithm,args.model,args.dataset,args.seed,args.percentage,args.layers_from_last,(sum(top_label_similarity_list)/len(top_label_similarity_list))/2]]
with open('src/data/log/overlap.csv', 'a') as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(csv_rows_each_round)


# select top k based on labels and compare mask overlap ratio

def calculate_label_affinity_list(labels,label_list):
    affinity_list = []
    n_client = len(label_list)

    for idx in range(n_client):
        label_similarity = len(set(labels)&set(label_list[idx]))
        affinity_list.append(label_similarity)
    
    return affinity_list

top_k_mask_ratio_list = []
overall_mask_ratio_list = []
for c_idx in range(n_client):
    affinity_list_based_on_mask_ratio = calculate_affinity_based_on_network(mask_list[c_idx],mask_list,args.n_conv_layer,args.layers_from_last)
    label_based_affinity_list = calculate_label_affinity_list(label_list[c_idx],label_list)

    # selected_idx = sorted(range(len(label_based_affinity_list)), key=lambda i: label_based_affinity_list[i])[-select_users_num:]
    selected_idx = []
    for ref_c_idx in range(n_client):
        if label_based_affinity_list[ref_c_idx] >= 1:
            selected_idx.append(ref_c_idx)
    
    print("base label",label_list[c_idx])

    for ref_c_idx in range(n_client):
        
        if c_idx == ref_c_idx:
            pass
        

        elif ref_c_idx in selected_idx:
            label_2 = label_list[ref_c_idx]

            print("selected label = ",label_2)

            top_k_mask_ratio_list.append(affinity_list_based_on_mask_ratio[ref_c_idx])
            overall_mask_ratio_list.append(affinity_list_based_on_mask_ratio[ref_c_idx])
        else:
            overall_mask_ratio_list.append(affinity_list_based_on_mask_ratio[ref_c_idx])

avg_all_mask_ratio = sum(overall_mask_ratio_list)/len(overall_mask_ratio_list)
avg_top_k_mask_ratio =    sum(top_k_mask_ratio_list)/len(top_k_mask_ratio_list)
print("average mask ratio overall",avg_all_mask_ratio)
print("average mask ratio top k ",avg_top_k_mask_ratio)
print("average mask ratio top k/average mask ratio overall ",avg_top_k_mask_ratio/avg_all_mask_ratio)

csv_rows_each_round = [[args.num_users,args.algorithm,args.model,args.dataset,args.seed,args.percentage,args.layers_from_last,"mask ratio of top k related clients",avg_all_mask_ratio,avg_top_k_mask_ratio,avg_top_k_mask_ratio/avg_all_mask_ratio]]
with open('src/data/log/overlap.csv', 'a') as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(csv_rows_each_round)


# select top k based on labels and compare weight divergence


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


        affinity_based_on_weight_divergence_list.append(- weight_divergence)

    return affinity_based_on_weight_divergence_list

final_weights_list = []
for c_idx in range(n_client):
    file_name_weights = "src/data/weights/" + str(args.algorithm) +"_"+str(args.model)+ "_" + str(args.num_users) + "_" + str(args.dataset) + "_" + str(args.seed) + "_client_id_" + str(c_idx) + ".pickle"
    file = open(file_name_weights, 'rb')

    # dump information to that file
    weights = pickle.load(file)

    weights_list = []
    for tensor in weights.items():
        
        weights_list.append(tensor[1])

    
    final_weights_list.append(weights_list)

top_k_network_similarity_list = []
overall_network_similarity_list = []
for c_idx in range(n_client):
    affinity_list_based_on_network_similarity = calculate_affinity_based_on_weight_divergence_of_locally_trained_models(final_weights_list,c_idx,args.n_conv_layer)
    label_based_affinity_list = calculate_label_affinity_list(label_list[c_idx],label_list)
    

    # selected_idx = sorted(range(len(label_based_affinity_list)), key=lambda i: label_based_affinity_list[i])[-select_users_num:]
    selected_idx = []
    for ref_c_idx in range(n_client):
        if label_based_affinity_list[ref_c_idx] >= 1:
            selected_idx.append(ref_c_idx)
    
    print("base label",label_list[c_idx])

    for ref_c_idx in range(n_client):
        
        if c_idx == ref_c_idx:
            pass
        

        elif ref_c_idx in selected_idx:
            label_2 = label_list[ref_c_idx]

            print("selected label = ",label_2)

            top_k_network_similarity_list.append(affinity_list_based_on_network_similarity[ref_c_idx])
            overall_network_similarity_list.append(affinity_list_based_on_network_similarity[ref_c_idx])
        else:
            overall_network_similarity_list.append(affinity_list_based_on_network_similarity[ref_c_idx])

avg_all_network_similarity= sum(overall_network_similarity_list)/len(overall_network_similarity_list)
avg_top_k_network_similarity =    sum(top_k_network_similarity_list)/len(top_k_network_similarity_list)
print("average network aimilarity overall",avg_all_network_similarity)
print("average network similarity top k ",avg_top_k_network_similarity)
print("average network similarity top k/average network similarity overall ",avg_all_network_similarity/avg_top_k_network_similarity)

csv_rows_each_round = [[args.num_users,args.algorithm,args.model,args.dataset,args.seed,args.percentage,args.layers_from_last,"network similarity of top k related clients",avg_all_network_similarity,avg_top_k_network_similarity,avg_all_network_similarity/avg_top_k_network_similarity]]
with open('src/data/log/overlap.csv', 'a') as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(csv_rows_each_round)


