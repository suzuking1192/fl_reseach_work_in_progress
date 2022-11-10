
import pickle
import copy
import numpy as np

dataset_list = ["mnist","cifar10","cifar100"]
# dataset_list = ["cifar10"]

mask_dict = {}
for dataset_name in dataset_list:

    file_name_mask = "src/data/masks/" + "ours" +"_"+ "lenet5" + "_" + "100"  + "_" + str(dataset_name) + "_" + "1_70_0.8_1.0_1.0" + "_client_id_" + "0" + ".pickle"
    
    
    # open a file, where you stored the pickled data
    file = open(file_name_mask, 'rb')

    # dump information to that file
    mask = pickle.load(file)

    mask_dict[dataset_name] = mask


def count_parameters(mask,n_conv_layer=2,all=True):
    n_layer = len(mask)

    total_num = 0
    alive = 0
    for l in range(n_layer):
        
        if l >= n_conv_layer:
            input_size = len(mask[l])
            output_size = len(mask[l][0])

            for i in range(input_size):
                for o in range(output_size):
                    if mask[l][i][o] == 1:
                        alive += 1
                    total_num += 1
                        
        else:
            # conv layer
            n_dim_0 = len(mask[l])
            n_dim_1 = len(mask[l][0])
            n_dim_2 = len(mask[l][0][0])
            n_dim_3 = len(mask[l][0][0][0])
            for i in range(n_dim_0):
                for j in range(n_dim_1):
                    for k in range(n_dim_2):
                        for m in range(n_dim_3):
                            if mask[l][i][j][k][m] == 1:
                                alive += 1
                            total_num += 1

    if all == True:
        return total_num
    else:
        return alive
                                

# communication cost analysis

## FedAVG

comm_rounds = 1000
bit_per_float = 32
pruning_per_round = 0.2
num_client_each_round = 10

for dataset_name in dataset_list:
    
    num_parameters = count_parameters(mask_dict[dataset_name],n_conv_layer=2,all=True)
    print("num_parameters",num_parameters)
    print("FedAVG communication cost",dataset_name,comm_rounds*bit_per_float*num_parameters*num_client_each_round*2*1.16*10**(-10),"GB")


for dataset_name in dataset_list:
    
    num_parameters_total = count_parameters(mask_dict[dataset_name],n_conv_layer=2,all=True)
    num_parameters_initial = copy.deepcopy(num_parameters_total)
    pruned_parameters_total = 0
    comm_cost_total = 0

    for iteration in range(comm_rounds):
        if iteration !=0 and iteration%10 ==0 and num_parameters_total > int(num_parameters_initial*0.3):
            num_parameters_total = num_parameters_total*(1-pruning_per_round)
            if num_parameters_total < int(num_parameters_initial*0.3):
                num_parameters_total = copy.deepcopy(int(num_parameters_initial*0.3))
            pruned_parameters_total = num_parameters_initial - num_parameters_total

        comm_cost_total += bit_per_float*num_parameters_total*num_client_each_round*1.16*10**(-10)*2 + pruned_parameters_total*num_client_each_round*1.16*10**(-10)*2
    print("num_parameters_initial =",num_parameters_initial)
    print("num_parameters_total =",num_parameters_total)
    print("Sub_FedAVG communication cost",dataset_name,comm_cost_total,"GB")


for dataset_name in dataset_list:
    num_parameters = count_parameters(mask_dict[dataset_name],n_conv_layer=2,all=True)

    comm_cost_total = comm_rounds*bit_per_float*int(num_parameters*0.3)*num_client_each_round*2*1.16*10**(-10) + comm_rounds*int(num_parameters*0.7)*num_client_each_round*2*1.16*10**(-10)

    print("num_parameters",num_parameters)
    print("DisPFL communication cost",dataset_name,comm_cost_total,"GB")

for dataset_name in dataset_list:
    
    num_parameters_total = count_parameters(mask_dict[dataset_name],n_conv_layer=2,all=True)
    num_parameters_initial = copy.deepcopy(num_parameters_total)
    pruned_parameters_total = 0
    comm_cost_total = 0

    delta_r = 10
    alpha = 0.2

    def cosine_annealing(alpha,iteration,T_end):
        return alpha / 2 * (1 + np.cos((iteration * np.pi) / T_end))

    for iteration in range(comm_rounds):
        if iteration !=0 and iteration%10 ==0 and num_parameters_total > int(num_parameters_initial*0.3):
            num_parameters_total = num_parameters_total*(1-pruning_per_round)
            if num_parameters_total < int(num_parameters_initial*0.3):
                num_parameters_total = copy.deepcopy(int(num_parameters_initial*0.3))

                if iteration%delta_r == 0:
                    alpha_decayed = cosine_annealing(alpha,iteration,comm_rounds)

                    num_parameters_total = num_parameters_total *(1+alpha_decayed)

            pruned_parameters_total = num_parameters_initial - num_parameters_total

        comm_cost_total += bit_per_float*num_parameters_total*num_client_each_round*1.16*10**(-10)*2 + pruned_parameters_total*num_client_each_round*1.16*10**(-10)*2
    
    # Communication cost for clustering
    num_client_total = 100
    comm_cost_total += bit_per_float*num_parameters_total*num_client_total*1.16*10**(-10)*2
    
    print("num_parameters_initial =",num_parameters_initial)
    print("num_parameters_total =",num_parameters_total)
    print("FedDNPR communication cost",dataset_name,comm_cost_total,"GB")


# FLOPs analysis

def flop_calculation(dataset,algorithm,n_conv_layer=2,all=False):
    if algorithm == "DisPFL":
        file_name_mask = "src/data/masks/" + str(algorithm) +"_"+"lenet5_100_"+ str(dataset)+ "_1_client_id_0.pickle"
    else:    
        file_name_mask = "src/data/masks/" + str(algorithm) +"_"+"lenet5" + "_" + "100" + "_" + str(dataset) + "_" + "1_70_0.8_1.0_1.0" + "_client_id_" + "0" + ".pickle"
    # open a file, where you stored the pickled data
    file = open(file_name_mask, 'rb')

    # dump information to that file
    mask = pickle.load(file)

    n_layer = len(mask)
    flop_total = 0
    flop_total_without_prune = 0
    for l in range(n_layer):
        if l < n_conv_layer:
            if dataset == "cifar10" and l ==0 :
                output_shape = 30*30
            elif dataset == "cifar10" and l ==1 :
                output_shape = 10*10
            elif dataset == "cifar100" and l ==0 :
                output_shape = 30*30
            elif dataset == "cifar100" and l ==1 :
                output_shape = 10*10
            elif dataset == "mnist" and l ==0 :
                output_shape = 28*28
            elif dataset == "mnist" and l ==1 :
                output_shape = 19*19
            output_dim = len(mask[l])
            input_dim = len(mask[l][0])
            filter_x = len(mask[l][0][0])
            filter_y = len(mask[l][0][0][0])

            flop_num = output_dim * (filter_x* filter_y) * output_shape
            flop_total += flop_num
            flop_total_without_prune += flop_num
        else:
            input_size = len(mask[l])
            output_size = len(mask[l][0])

            flop_num = 0

            for i in range(input_size):
                for o in range(output_size):
                    if mask[l][i][o] == 1:
                        flop_num += 1
                    flop_total_without_prune += 1
            flop_total += copy.deepcopy(flop_num)
    if all == False:
        return flop_total
    else:
        return flop_total_without_prune

for dataset_name in dataset_list:
    flops = flop_calculation(dataset_name,"sub_fedavg",n_conv_layer=2)
    print("SUb-FedAVG FLOPs",dataset_name,flops)

for dataset_name in dataset_list:
    flops = flop_calculation(dataset_name,"ours",n_conv_layer=2)
    print("FedPMS FLOPs",dataset_name,flops)

for dataset_name in dataset_list:
    flops = flop_calculation(dataset_name,"ours",n_conv_layer=2,all=True)
    print("FedAVG FLOPs",dataset_name,flops)

for dataset_name in dataset_list:
    flops = flop_calculation(dataset_name,"DisPFL",n_conv_layer=2)
    print("DisPFL FLOPs",dataset_name,flops)

