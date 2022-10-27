
import numpy as np
import random
import copy
import statistics
import torch
from src.sub_fedavg import * 
import csv
from scipy.spatial import distance
import pickle
import math

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


def create_selected_client_idx_list(affinity_mat,parameter_to_multiply_avg):

    # calculate average of affinity measure
    affinity_measure_list = []
    n_client = len(affinity_mat)

    for c_idx in range(n_client):
        for compared_idx in range(n_client):
            if c_idx != compared_idx:
                affinity_measure_list.append(affinity_mat[c_idx][compared_idx])
    
    avg_affinity_measure = sum(affinity_measure_list)/len(affinity_measure_list)
    adjusted_threshold = avg_affinity_measure * parameter_to_multiply_avg


    # select similar clients
    selected_idxs = []
    for c_idx in range(n_client):
        selected_idxs.append([])
        for compared_idx in range(n_client):
            if c_idx != compared_idx:
                if affinity_mat[c_idx][compared_idx] > adjusted_threshold: # affinity measure is negative valuse because it is multiples by -1 with weight divergence
                    selected_idxs[c_idx].append(compared_idx)

    return selected_idxs

def create_selected_idx_mat(clients,n_conv_layer,parameter_to_multiply_avg):
    n_client = len(clients)
    # local training
    locally_trained_weights_list = []
    for c_idx in range(n_client):
        clients[c_idx].fake_train()

        weights_list = []
        for tensor in clients[c_idx].get_fake_state_dict().items():
            
            weights_list.append(tensor[1])

        
        locally_trained_weights_list.append(weights_list)

    # create affinity matrix
    affinity_mat = []
    for c_idx in range(n_client):
        affinty_list = calculate_affinity_based_on_weight_divergence_of_locally_trained_models(locally_trained_weights_list,c_idx,n_conv_layer)
        affinity_mat.append(affinty_list)

    # Create selected idx list
    selected_idx_mat = create_selected_client_idx_list(affinity_mat,parameter_to_multiply_avg)

    return selected_idx_mat

def regrowth_based_on_affinity_c_idxs(binary_mask,binary_mask_list_all,selected_client_idx,mask_readjustment_rate,n_conv_layer):
    mask = copy.deepcopy(binary_mask)
    
    n_layer = len(binary_mask)

    counter = 0
    overlap = 0 
    for l in range(n_layer):
        if l < n_conv_layer:
            # conv layer
            n_dim_0 = len(mask[l])
            n_dim_1 = len(mask[l][0])
            n_dim_2 = len(mask[l][0][0])
            n_dim_3 = len(mask[l][0][0][0])
            for i in range(n_dim_0):
                for j in range(n_dim_1):
                    for k in range(n_dim_2):
                        for m in range(n_dim_3):
                            if mask[l][i][j][k][m] == 0:
                                counter += 1
                                overlap_status = False
                                for c_idx in selected_client_idx:
                                    if binary_mask_list_all[c_idx][l][i][j][k][m] == 1:
                                        overlap_status = True
                                if overlap_status == True:
                                    overlap += 1
        else:
            # FC layer
            input_size = len(mask[l])
            output_size = len(mask[l][0])

            for i in range(input_size):
                for o in range(output_size):
                    
                    if mask[l][i][o] == 0:
                        counter += 1
                        overlap_status = False
                        for c_idx in selected_client_idx:
                            if binary_mask_list_all[c_idx][l][i][o] == 1:
                                overlap_status = True
                        if overlap_status == True:
                            overlap += 1

                    
    # randomly select necessary number of indexes
    if int(counter*mask_readjustment_rate) > overlap-1:
        idx_list = range(overlap)
    else:
        idx_list = random.sample(range(overlap-1), int(counter*mask_readjustment_rate))

    

    # Assign 1 to those mask
    idx_counter = 0
    for l in range(n_layer):
        if l < n_conv_layer:
            # conv layer
            n_dim_0 = len(mask[l])
            n_dim_1 = len(mask[l][0])
            n_dim_2 = len(mask[l][0][0])
            n_dim_3 = len(mask[l][0][0][0])
            for i in range(n_dim_0):
                for j in range(n_dim_1):
                    for k in range(n_dim_2):
                        for m in range(n_dim_3):
                            if mask[l][i][j][k][m] == 0:
                                overlap_status = False
                                for c_idx in selected_client_idx:
                                    if binary_mask_list_all[c_idx][l][i][j][k][m] == 1:
                                        overlap_status = True
                                if overlap_status == True:
                                    if idx_counter in idx_list:
                                        mask[l][i][j][k][m] = 1
                                    

                                    idx_counter += 1

        else:
            # FC layer
            input_size = len(mask[l])
            output_size = len(mask[l][0])

            for i in range(input_size):
                for o in range(output_size):
                    if mask[l][i][o] == 0:
                        overlap_status = False
                        for c_idx in selected_client_idx:
                            if binary_mask_list_all[c_idx][l][i][o] == 1:
                                overlap_status = True
                        if overlap_status == True:
                            if idx_counter in idx_list:
                                mask[l][i][o] = 1
                            

                            idx_counter += 1

    return mask

def model_growing(mask,mask_adjustment_rate,n_conv_layer):
    updated_mask = copy.deepcopy(mask)
    # Count total number of pruned weights
    n_layer = len(mask)

    counter = 0
    total_count = 0
    for l in range(n_layer):
        if l < n_conv_layer:
            # conv layer
            n_dim_0 = len(mask[l])
            n_dim_1 = len(mask[l][0])
            n_dim_2 = len(mask[l][0][0])
            n_dim_3 = len(mask[l][0][0][0])
            for i in range(n_dim_0):
                for j in range(n_dim_1):
                    for k in range(n_dim_2):
                        for m in range(n_dim_3):
                            total_count += 1
                            if mask[l][i][j][k][m] == 0:
                                counter += 1
        else:
            # FC layer
            input_size = len(mask[l])
            output_size = len(mask[l][0])

            for i in range(input_size):
                for o in range(output_size):
                    total_count += 1
                    if mask[l][i][o] == 0:
                        counter += 1
                    


    # randomly select necessary number of indexes
    idx_list = random.sample(range(counter-1), int(counter*mask_adjustment_rate))

    

    # Assign 1 to those mask
    idx_counter = 0
    for l in range(n_layer):
        if l < n_conv_layer:
            # conv layer
            n_dim_0 = len(mask[l])
            n_dim_1 = len(mask[l][0])
            n_dim_2 = len(mask[l][0][0])
            n_dim_3 = len(mask[l][0][0][0])
            for i in range(n_dim_0):
                for j in range(n_dim_1):
                    for k in range(n_dim_2):
                        for m in range(n_dim_3):
                            if mask[l][i][j][k][m] == 0:
                                if idx_counter in idx_list:
                                    updated_mask[l][i][j][k][m] = 1
                                    

                                idx_counter += 1
        else:
            # FC layer
            input_size = len(mask[l])
            output_size = len(mask[l][0])

            for i in range(input_size):
                for o in range(output_size):
                    if mask[l][i][o] == 0:
                        if idx_counter in idx_list:
                            updated_mask[l][i][o] = 1
                            

                        idx_counter += 1

    updated_pruned_rate = (counter - len(idx_list))/total_count

    next_prune_rate = len(idx_list)/(total_count - counter + len(idx_list))

    return updated_mask,updated_pruned_rate*100,next_prune_rate


def global_multi_criteria_pruning(weights_list_with_pruning_status,mask_list,lambda_value,pruned_rate_each_round,n_conv_layer,pruned_rate_list,pruned_target):
    updated_mask_list = []
    updated_pruned_rate_list = []

    n_client = len(mask_list)
    n_layer = len(mask_list[0])
    for c_id in range(n_client):
        weights = copy.deepcopy(weights_list_with_pruning_status[c_id][0])
        mask = copy.deepcopy(mask_list[c_id])
        
        if weights_list_with_pruning_status[c_id][1] == False:
            updated_mask_list.append(mask)
            updated_pruned_rate_list.append(pruned_rate_list[c_id])
        else:

            # Normalize weights and standard deviation loss
            weights_list = []
            std_weights_list = []
            std_weights_list_multi_dim = []
            unpruned = 0
            num_weights = 0

            for l in range(n_layer):
                std_weights_list_multi_dim.append([])
                if l < n_conv_layer:
                    # conv layer
                    n_dim_0 = len(mask[l])
                    n_dim_1 = len(mask[l][0])
                    n_dim_2 = len(mask[l][0][0])
                    n_dim_3 = len(mask[l][0][0][0])
                    for i in range(n_dim_0):
                        std_weights_list_multi_dim[l].append([])
                        for j in range(n_dim_1):
                            std_weights_list_multi_dim[l][i].append([])
                            for k in range(n_dim_2):
                                std_weights_list_multi_dim[l][i][j].append([])
                                for m in range(n_dim_3):
                                    weights_list.append(weights[l*2][i][j][k][m])
                                    num_weights += 1

                                    if mask[l][i][j][k][m] != 0 :
                                        unpruned +=1
                                    
                                    # Calculate std
                                    if mask[l][i][j][k][m] == 0:
                                        std_weights_list.append(0)
                                        std_weights_list_multi_dim[l][i][j][k].append(0)
                                    else:
                                        weight_list_in_one_position = []
                                        weight_list_in_one_position_without_ref_weight = []
                                        for c_idx in range(n_client):
                                            if mask_list[c_idx][l][i][j][k][m] != 0:
                                                if c_id == c_idx:
                                                    weight_list_in_one_position.append(weights_list_with_pruning_status[c_idx][0][l*2][i][j][k][m].item())
                                                else:
                                                    weight_list_in_one_position_without_ref_weight.append(weights_list_with_pruning_status[c_idx][0][l*2][i][j][k][m].item())
                                                    weight_list_in_one_position.append(weights_list_with_pruning_status[c_idx][0][l*2][i][j][k][m].item())
                                        if (len(weight_list_in_one_position) == 0) or (len(weight_list_in_one_position_without_ref_weight) == 0):
                                            std_diff = 0
                                        else:
                                            std_diff = abs(statistics.pstdev(weight_list_in_one_position) - statistics.pstdev(weight_list_in_one_position_without_ref_weight))
                                        if std_diff == 0:
                                            std_weights_list.append(std_diff)
                                            std_weights_list_multi_dim[l][i][j][k].append(std_diff)
                                        else:
                                            std_weights_list.append(std_diff)
                                            std_weights_list_multi_dim[l][i][j][k].append(std_diff)


                else:
                    input_size = len(mask[l])
                    output_size = len(mask[l][0])

                    for i in range(input_size):
                        std_weights_list_multi_dim[l].append([])
                        for o in range(output_size):
                            
                            weights_list.append(weights[l*2][i][o])

                            num_weights += 1
                            if mask[l][i][o] != 0:
                                unpruned +=1

                            # Calculate std
                            if mask[l][i][o] == 0:
                                std_weights_list.append(0)
                                std_weights_list_multi_dim[l][i].append(0)
                            else:

                                weight_list_in_one_position = []
                                weight_list_in_one_position_without_ref_weight = []
                                for c_idx in range(n_client):
                                    if mask_list[c_idx][l][i][o] != 0:
                                        if c_id == c_idx:
                                            weight_list_in_one_position.append(weights_list_with_pruning_status[c_idx][0][l*2][i][o].item())
                                        else:
                                            weight_list_in_one_position_without_ref_weight.append(weights_list_with_pruning_status[c_idx][0][l*2][i][o].item())
                                            weight_list_in_one_position.append(weights_list_with_pruning_status[c_idx][0][l*2][i][o].item())
                                if (len(weight_list_in_one_position) == 0) or (len(weight_list_in_one_position_without_ref_weight) == 0):
                                    std_diff = 0
                                else:
                                    std_diff = abs(statistics.pstdev(weight_list_in_one_position) - statistics.pstdev(weight_list_in_one_position_without_ref_weight))
                                if std_diff == 0:
                                    std_weights_list.append(std_diff)
                                    std_weights_list_multi_dim[l][i].append(std_diff)
                                else:
                                    std_weights_list.append(std_diff)
                                    std_weights_list_multi_dim[l][i].append(std_diff)
            
            sum_weights = sum(weights_list)
            sum_std_weights = sum(std_weights_list)
            # print("sum_std_weights =",sum_std_weights)
            # print("statistics of values------------------------")
            # print("mean of weights= ", statistics.mean(weights_list))
            # print("mean of std = ",statistics.mean(std_weights_list))
            # print("maximum of weights =",max(weights_list))
            # print("maximum of std=",max(std_weights_list))
            # print("std of weights =",statistics.stdev(weights_list))
            # print("std of std=",statistics.stdev(std_weights_list))

            # Add values and sort them
            all_weights = {}
            counter = 0
            for l in range(n_layer):
                if l < n_conv_layer:
                    # conv layer
                    n_dim_0 = len(mask[l])
                    n_dim_1 = len(mask[l][0])
                    n_dim_2 = len(mask[l][0][0])
                    n_dim_3 = len(mask[l][0][0][0])
                    for i in range(n_dim_0):
                        for j in range(n_dim_1):
                            for k in range(n_dim_2):
                                for m in range(n_dim_3):
                                    try:
                                        value = abs(weights[l*2][i][j][k][m]/sum_weights) - lambda_value * std_weights_list_multi_dim[l][i][j][k][m]/sum_std_weights
                                    except:
                                        value = abs(weights[l*2][i][j][k][m]/sum_weights)
                                    all_weights[(l,i,j,k,m)] = value
                                    counter += 1
             
                else:
                    input_size = len(mask[l])
                    output_size = len(mask[l][0])

                    for i in range(input_size):
                        for o in range(output_size):
                            
                            try:
                                # if (i==0) and (o == 0):
                                    # print("weights[l*2][i][o]",abs(weights[l*2][i][o]/sum_weights))
                                    # print("std_weights_list_multi_dim[l][i][o]",std_weights_list_multi_dim[l][i][o]/sum_std_weights)
                            
                                value = abs(weights[l*2][i][o]/sum_weights) - lambda_value * std_weights_list_multi_dim[l][i][o]/sum_std_weights
                            except:
                                if (i==0) and (o == 0):
                                    print("division by zero")
                                value = abs(weights[l*2][i][o]/sum_weights)
                            all_weights[(l,i,o)] = value
                            counter += 1

            all_weights_sorted = {k: v for k, v in sorted(all_weights.items(), key=lambda item: abs(item[1]))}
            # Prune weights


            pruning_percent = 1 -  (unpruned/num_weights) * (1-pruned_rate_each_round/100)
            if pruning_percent > pruned_target/100:
                pruning_percent = copy.deepcopy(pruned_target/100)
            # print("unpruned/num_weights",unpruned/num_weights)
            # print("pruning_percent =",pruning_percent)

            number_of_weights_to_be_pruned = int(pruning_percent*num_weights)
            # print("number_of_weights_to_be_pruned= ",number_of_weights_to_be_pruned)
            weights_to_be_pruned = {k: all_weights_sorted[k] for k in list(all_weights_sorted)[ :  number_of_weights_to_be_pruned]}     

            
            for k, v in weights_to_be_pruned.items():
                if k[0] >= n_conv_layer: # fully connected layer
                    
                    mask[k[0]][k[1]][k[2]] = 0
                    
                else:
                    
                    mask[k[0]][k[1]][k[2]][k[3]][k[4]] = 0
            
            updated_mask_list.append(mask)
            updated_pruned_rate_list.append(pruning_percent*100)
            mask_list[c_id] = copy.deepcopy(mask)

            


    return updated_mask_list,updated_pruned_rate_list

def fill_zero_weights(state_dict,n_conv_layer,lr = None,layer_wise=False):
    weights_list = []
    n_layer = int(len(state_dict)/2)

    weights = []
    for tensor in state_dict.items():
        weights.append(tensor[1])

    if layer_wise == True:
        layer_wise_weights_list = []

    if lr == None:
        for l in range(n_layer):
            if layer_wise == True:
                layer_wise_weights_list.append([])
            if l < n_conv_layer:
                # conv layer
                n_dim_0 = len(weights[l*2])
                n_dim_1 = len(weights[l*2][0])
                n_dim_2 = len(weights[l*2][0][0])
                n_dim_3 = len(weights[l*2][0][0][0])
                for i in range(n_dim_0):
                    for j in range(n_dim_1):
                        for k in range(n_dim_2):
                            for m in range(n_dim_3):
                                if weights[l*2][i][j][k][m] == 0:
                                    #weights[l*2][i][o] = np.random.normal(0, 0.001, 1)[0]
                                    pass
                                else:
                                    weights_list.append(weights[l*2][i][j][k][m].item())
                                    if layer_wise == True:
                                        layer_wise_weights_list[l].append(weights[l*2][i][j][k][m].item())
            else:
                n_neurons_layer = weights[l*2].shape[0]
                n_output = weights[l*2].shape[1]

                for i in range(n_neurons_layer):
                    for o in range(n_output):
                        if weights[l*2][i][o] == 0:
                            #weights[l*2][i][o] = np.random.normal(0, 0.001, 1)[0]
                            pass
                        else:
                            weights_list.append(weights[l*2][i][o].item())
                            if layer_wise == True:
                                layer_wise_weights_list[l].append(weights[l*2][i][o].item())
        res = statistics.pstdev(weights_list)
        # Printing result
        # print("Standard deviation of weights is : " + str(res))
        if layer_wise == True:
            res_layer = []
            for l in range(n_layer):
                print("layer = ",l)
                res = statistics.pstdev(layer_wise_weights_list[l])
                res_layer.append(res)
                # Printing result
                print("Standard deviation of weights is : " + str(res))
                

    else:
        res = lr

    for tensor in state_dict.items():
        if "weight" in tensor[0]:
            weight_list = []
            try:
                n_dim_0 = len(tensor[1])
                n_dim_1 = len(tensor[1][0])
                n_dim_2 = len(tensor[1][0][0])
                n_dim_3 = len(tensor[1][0][0][0])
                for i in range(n_dim_0):
                    weight_list.append([])
                    for j in range(n_dim_1):
                        weight_list[i].append([])
                        for k in range(n_dim_2):
                            weight_list[i][j].append([])
                            for m in range(n_dim_3):
                                if tensor[1][i][j][k][m] == 0:
                                    if layer_wise == False:
                                        weight_list[i][j][k].append(np.random.normal(0, res, 1)[0])
                                    else:
                                        weight_list[i][j][k].append(np.random.normal(0, res_layer[l], 1)[0])
                                else:
                                    weight_list[i][j][k].append(tensor[1][i][j][k][m].item())
                weight_array = np.array(weight_list)
                weight_tensor = torch.from_numpy(weight_array)
                state_dict[str(tensor[0])] = weight_tensor
                
            except:
                n_dim_0 = len(tensor[1])
                n_dim_1 = len(tensor[1][0])
                for i in range(n_dim_0):
                    weight_list.append([])
                    for j in range(n_dim_1):
                        
                        if tensor[1][i][j] == 0:
                            if layer_wise == False:
                                weight_list[i].append(np.random.normal(0, res, 1)[0])
                            else:
                                weight_list[i].append(np.random.normal(0, res_layer[l], 1)[0])
                        else:
                            weight_list[i].append(tensor[1][i][j].item())
                weight_array = np.array(weight_list)
                weight_tensor = torch.from_numpy(weight_array)
                state_dict[str(tensor[0])] = weight_tensor
    


    return state_dict


def weighted_global_model_update(server_state_dict,local_averaged_server_state_dict,frac):
    for key,tensor in server_state_dict.items():
        server_state_dict[key] =  (1/1+frac) * tensor + (frac/(1+frac)) * local_averaged_server_state_dict[key]

    return server_state_dict


def create_data_to_learn_positive_transfer(clients,args,initial_state_dict,num_client,users_train_labels,train_dataset,test_dataset,user_groups_train,training_round=20,bias=True,total=True,labels=True,distance_change_over_training=50,cosine=True,prediction_dist=True,val_acc_another_model=True):
    # Local training
    locally_trained_weights_list  = []
    for c_idx in range(num_client):
        clients[c_idx].local_train()
        locally_trained_weights_list.append(clients[c_idx].get_state_dict())
    
        # Convert tensor into numpy arrays, then save them
        local_weights_tensor = clients[c_idx].get_state_dict()
        local_weights_array = []

        for tensor in local_weights_tensor.items():
            local_weights_array.append(tensor[1].numpy())
        
        local_weights_array = np.array(local_weights_array)

        numpy_filename = "src/data/weights/local_weights_array_for_positive_transfer_training_" + str(c_idx) +".npy" 
        with open(numpy_filename, 'wb') as f:
            np.save(f, local_weights_array)
    
    # Calculate initial accuracy
    init_acc_list = []
    
    for c_idx in range(num_client):
        init_val_acc_max = 0
        for i_round in range(training_round):
            clients[c_idx].local_train()
            loss_c,init_val_acc_new = clients[c_idx].eval_val()
            if init_val_acc_max < init_val_acc_new:
                init_val_acc_max = init_val_acc_new
        init_acc_list.append(init_val_acc_max)


    # For each possible pair, first calculate l2 distance of each layer and save it
    # Then, create average parameters and local train and save average of validation accuracy
    l2_dis_list = []
    positive_transfer_list = []
    counter = 0
    for c_idx in range(num_client):
        for ref_idx in range(num_client):
            if c_idx != ref_idx:
                clients[c_idx].set_state_dict(locally_trained_weights_list[c_idx])
                clients[ref_idx].set_state_dict(locally_trained_weights_list[ref_idx])
                init_val_acc_c = init_acc_list[c_idx]
                init_val_acc_ref = init_acc_list[ref_idx]
                loss_c,init_val_acc_c_first_round = clients[c_idx].eval_val()
                loss_c,init_val_acc_ref_first_round = clients[ref_idx].eval_val()
                
                l2_dis_list.append([])

                

                for tensor in locally_trained_weights_list[c_idx].items():
                    

                    if cosine == True:
                        
                        weight_1 = torch.flatten(tensor[1] - initial_state_dict[tensor[0]])
                        weight_2 = torch.flatten(locally_trained_weights_list[ref_idx][tensor[0]] - initial_state_dict[tensor[0]])
                        


                        # cosine_sim = distance.cosine(weight_1.tolist(), weight_2.tolist())
                        # cosine_sim = torch.sum(weight_1*weight_2)/(torch.norm(weight_1)*torch.norm(weight_2)+1e-12)
                        
                        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                        
                        cosine_sim = cos(weight_1, weight_2)
                        
                        l2_dis_list[counter].append(cosine_sim.detach().numpy().item())
                    else:
                        if "weight" in tensor[0]:
                            if "conv" in tensor[0]:
                                l2_dist = (tensor[1] - locally_trained_weights_list[ref_idx][tensor[0]]).pow(2).sum(3).sum(1).sum(1).sum().sqrt()  
                            else:
                                l2_dist = (tensor[1] - locally_trained_weights_list[ref_idx][tensor[0]]).pow(2).sum(1).sum().sqrt()  
                            l2_dis_list[counter].append(l2_dist.detach().numpy().item())
                        elif bias == True:
                            
                            l2_dist = (tensor[1] - locally_trained_weights_list[ref_idx][tensor[0]]).pow(2).sum().sqrt()  
                            l2_dis_list[counter].append(l2_dist.detach().numpy().item())
                
                if val_acc_another_model == True:

                    clients[c_idx].set_state_dict(locally_trained_weights_list[ref_idx])
                    clients[ref_idx].set_state_dict(locally_trained_weights_list[c_idx])
                    loss_c,init_val_acc_switch_c = clients[c_idx].eval_val()
                    loss_c,init_val_acc_sqitch_ref = clients[ref_idx].eval_val()
                    avg_val_switch_acc = ((init_val_acc_switch_c - init_val_acc_c) + (init_val_acc_sqitch_ref - init_val_acc_ref) )/2
                    l2_dis_list[counter].append(avg_val_switch_acc.detach().numpy().item())

                    clients[c_idx].set_state_dict(locally_trained_weights_list[c_idx])
                    clients[ref_idx].set_state_dict(locally_trained_weights_list[ref_idx])

                
                if labels == True:
                    similarity = len(set(users_train_labels[c_idx]) & set(users_train_labels[ref_idx])) 
                    l2_dis_list[counter].append(similarity)
                
                if total == True:
                    if cosine == True:
                        counter_tensor = 0
                        for tensor in locally_trained_weights_list[c_idx].items():
                            if "weight" in tensor[0]:
                                if counter_tensor == 0:
                                    weight_1_all = torch.flatten(tensor[1] - initial_state_dict[tensor[0]])
                                    counter_tensor += 1
                                else:
                                    weight_1_all = torch.cat((weight_1, torch.flatten(tensor[1] - initial_state_dict[tensor[0]])), 0)
                        counter_tensor = 0
                        for tensor in locally_trained_weights_list[ref_idx].items():
                            if "weight" in tensor[0]:
                                if counter_tensor == 0:
                                    weight_2_all = torch.flatten(tensor[1]- initial_state_dict[tensor[0]])
                                    counter_tensor += 1
                                else:
                                    weight_2_all = torch.cat((weight_2, torch.flatten(tensor[1]- initial_state_dict[tensor[0]])), 0)

                        # weight_1 = torch.flatten(tensor[1])
                        # weight_2 = torch.flatten(locally_trained_weights_list[ref_idx][tensor[0]])
                        # cosine_sim = torch.sum(weight_1*weight_2)/(torch.norm(weight_1)*torch.norm(weight_2)+1e-12)
                        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                        cosine_sim = cos(weight_1_all, weight_2_all)
                        

                        l2_dis_list[counter].append(cosine_sim.detach().numpy().item())

                    else:
                        num_weights = len(l2_dis_list[counter])
                        sum = 0 
                        for i in range(num_weights):
                            sum += l2_dis_list[counter][i]
                        l2_dis_list[counter].append(sum)

                if prediction_dist == True:
                    
                    n_data = 10
                    
                    model_1 = clients[c_idx].get_net()
                    model_2 = clients[ref_idx].get_net()
                    for i in range(n_data):
                        if i == 0:
                            pred_1 = model_1(test_dataset[i][0])
                            pred_2 = model_2(test_dataset[i][0])
                        else:
                            pred_1 = torch.cat((pred_1,model_1(test_dataset[i][0])),0)
                            pred_2 = torch.cat((pred_2,model_2(test_dataset[i][0])),0)

                        
                    
                    loss = nn.CrossEntropyLoss()
                    

                    prediction_cross_entropy = loss(pred_1, pred_2)
                    


                    # prediction_dist = (pred_1 - pred_2).pow(2).sum(1).sum().sqrt()  
                    l2_dis_list[counter].append(prediction_cross_entropy.detach().numpy().item())
                
                # Create an average global model
                for i_round in range(training_round):
                    masks = []
                    w_locals = []
                    masks.append(copy.deepcopy(clients[c_idx].get_mask())) 
                    masks.append(copy.deepcopy(clients[ref_idx].get_mask()))         
                    w_locals.append(copy.deepcopy(clients[c_idx].get_state_dict()))
                    w_locals.append(copy.deepcopy(clients[ref_idx].get_state_dict()))

                    avg_server_state_dict = Sub_FedAVG_U(initial_state_dict, w_locals, masks)

                    clients[c_idx].set_state_dict(avg_server_state_dict)
                    clients[ref_idx].set_state_dict(avg_server_state_dict)

                    clients[c_idx].local_train()
                    clients[ref_idx].local_train()
                    if i_round < distance_change_over_training:
                        dist_sum = 0
                        for tensor in clients[c_idx].get_state_dict().items():
                            if "weight" in tensor[0]:
                                if "conv" in tensor[0]:
                                    l2_dist = (tensor[1] - clients[ref_idx].get_state_dict()[tensor[0]]).pow(2).sum(3).sum(1).sum(1).sum().sqrt()  
                                else:
                                    l2_dist = (tensor[1] - clients[ref_idx].get_state_dict()[tensor[0]]).pow(2).sum(1).sum().sqrt()  
                                dist_sum += l2_dist.detach().numpy().item()
                            elif bias == True:
                                
                                l2_dist = (tensor[1] - clients[ref_idx].get_state_dict()[tensor[0]]).pow(2).sum().sqrt()  
                                dist_sum += l2_dist.detach().numpy().item()

                        change_dist = l2_dis_list[counter][-1-i_round*2] - dist_sum
                        l2_dis_list[counter].append(change_dist)

                        loss_c,middle_val_acc_c = clients[c_idx].eval_val()
                        loss_c,middle_val_acc_ref = clients[ref_idx].eval_val()
                        avg_val_acc = ((middle_val_acc_c - init_val_acc_c_first_round) + (middle_val_acc_ref - init_val_acc_ref_first_round) )/2
                        l2_dis_list[counter].append(avg_val_acc.detach().numpy().item())

                    if i_round == 0:
                        loss_c,val_acc_c = clients[c_idx].eval_val()
                        loss_ref,val_acc_ref = clients[ref_idx].eval_val()
                    else:
                        loss_c,val_middle_acc_c = clients[c_idx].eval_val()
                        loss_ref,val_middle_acc_ref = clients[ref_idx].eval_val()

                        if val_middle_acc_c > val_acc_c:
                            val_acc_c = val_middle_acc_c
                        if val_middle_acc_ref > val_acc_ref:
                            val_acc_ref = val_middle_acc_ref
                        print("val_middle_acc_c,val_middle_acc_ref =",val_middle_acc_c,val_middle_acc_ref)
                

                avg_val_acc = ((val_acc_c - init_val_acc_c) + (val_acc_ref - init_val_acc_ref) )/2
                positive_transfer_list.append(avg_val_acc.detach().numpy().item())
                counter += 1
    if args.model == "lenet5":
        file_name_common = "src/data/positive_transfer/dataset_" + str(args.dataset) +"num_user" + str(args.num_users) + "training_round_" + str(training_round) + "cosine_" +str(cosine)
    else:
        file_name_common = "src/data/positive_transfer/dataset_"  + str(args.dataset)+ "_model_"+ str(args.model) +"num_user" + str(args.num_users) + "training_round_" + str(training_round) + "cosine_" +str(cosine)

    sim_file_name = file_name_common + "similarity_measure.csv"
    with open(sim_file_name, 'w') as f:
    # using csv.writer method from CSV package
        write = csv.writer(f)
        
        write.writerows(l2_dis_list)

    pt_file_name = file_name_common + "positive_transfer_list.csv"
    with open(pt_file_name, 'w') as f:
    # using csv.writer method from CSV package
        write = csv.writer(f)
        
        write.writerows([positive_transfer_list])

    
    
    # with open("training_data_ids.pickle", 'wb') as fp:
    #     pickle.dump(user_groups_train, fp)

    # with open("training_data.pickle", 'wb') as fp:
    #     pickle.dump(train_dataset, fp)


def k_means_clustering(clients,initial_state_dict,n_cluster,n_layer,return_cluster_ids=False):
    
    # Create an array for gradients
    
    output_layer_gradient_list = []
    n_client = len(clients)

    for c_idx in range(n_client):
        clients[c_idx].fake_train()
        tensor_dict = clients[c_idx].get_fake_state_dict()
        counter_tensor = 0
        

        for tensor in tensor_dict.items():
            if "weight" in tensor[0]:
                if counter_tensor == n_layer-1:
                    weight = torch.flatten(tensor[1] - initial_state_dict[tensor[0]])
                counter_tensor += 1
                
        output_layer_gradient_list.append(weight.detach().numpy())

    

    gradients_array = np.array(output_layer_gradient_list)
    # Calculate k-means based on cosine similarity
    from sklearn import preprocessing  # to normalise existing X
    

    X_Norm = preprocessing.normalize(gradients_array)
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(X_Norm)
    if return_cluster_ids == True:
        return kmeans.labels_

    # Create a matrix
    similar_client_matrix = []
    for c_idx in range(n_client):
        similar_client_matrix.append([])
        for ref_idx in range(n_client):
            if c_idx != ref_idx:
                if kmeans.labels_[c_idx] == kmeans.labels_[ref_idx]:
                    similar_client_matrix[c_idx].append(ref_idx)

    return similar_client_matrix

def calculate_sparsities(params, tabu=[], distribution="ERK", sparse = 0.5):
    spasities = {}
    if distribution == "uniform":
        for name in params:
            if name not in tabu:
                spasities[name] = 1 - dense_ratio
            else:
                spasities[name] = 0
    elif distribution == "ERK":
        
        total_params = 0
        for name in params:
            total_params += params[name].numel()
        is_epsilon_valid = False
        # # The following loop will terminate worst case when all masks are in the
        # custom_sparsity_map. This should probably never happen though, since once
        # we have a single variable or more with the same constant, we have a valid
        # epsilon. Note that for each iteration we add at least one variable to the
        # custom_sparsity_map and therefore this while loop should terminate.
        dense_layers = set()

        density = sparse
        while not is_epsilon_valid:
            # We will start with all layers and try to find right epsilon. However if
            # any probablity exceeds 1, we will make that layer dense and repeat the
            # process (finding epsilon) with the non-dense layers.
            # We want the total number of connections to be the same. Let say we have
            # for layers with N_1, ..., N_4 parameters each. Let say after some
            # iterations probability of some dense layers (3, 4) exceeded 1 and
            # therefore we added them to the dense_layers set. Those layers will not
            # scale with erdos_renyi, however we need to count them so that target
            # paratemeter count is achieved. See below.
            # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
            #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
            # eps * (p_1 * N_1 + p_2 * N_2) =
            #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
            # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name in params:
                if name in tabu:
                    dense_layers.add(name)
                n_param = np.prod(params[name].shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if name in dense_layers:
                    rhs -= n_zeros
                else:
                    rhs += n_ones
                    raw_probabilities[name] = (
                                                        np.sum(params[name].shape) / np.prod(params[name].shape)
                                                ) 
                    divisor += raw_probabilities[name] * n_param
            epsilon = rhs / divisor
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        (f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name in params:
            if name in dense_layers:
                spasities[name] = 0
            else:
                spasities[name] = (1 - epsilon * raw_probabilities[name])
    return spasities


def init_masks(params, sparsities):
    masks ={}
    for name,tensor in params.items():
        masks[name] = torch.zeros_like(tensor)
        dense_numel = int((1-sparsities[name])*torch.numel(masks[name]))
        if dense_numel > 0:
            temp = masks[name].view(-1)
            perm = torch.randperm(len(temp))
            perm = perm[:dense_numel]
            temp[perm] =1
    return masks

def fire_mask(masks, weights,device ,round,comm_round,mask_regularization,anneal_factor):
    drop_ratio = anneal_factor / 2 * (1 + np.cos((round * np.pi) / comm_round))
    new_masks = copy.deepcopy(masks)

    num_remove = {}
    for name in masks:
        

        num_non_zeros = torch.sum(masks[name])
        num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
        temp_weights = torch.where(masks[name] > 0, torch.abs(weights[name]+mask_regularization[name]), 100000 * torch.ones_like(weights[name]))
        x, idx = torch.sort(temp_weights.view(-1).to(device))
        new_masks[name].view(-1)[idx[:num_remove[name]]] = 0
    return new_masks, num_remove


# we only update the private components of client's mask
def regrow_mask(masks,  num_remove,device ,mask_regularization,gradient=None,dis_gradient_check=None):
    new_masks = copy.deepcopy(masks)
    for name in masks:
        # if name not in public_layers:
            # if "conv" in name:
            if not dis_gradient_check:
                temp = torch.where(masks[name] == 0, torch.abs(gradient[name]+mask_regularization[name]), -100000 * torch.ones_like(gradient[name]))
                sort_temp, idx = torch.sort(temp.view(-1).to(device), descending=True)
                new_masks[name].view(-1)[idx[:num_remove[name]]] = 1
            else:
                temp = torch.where(masks[name] == 0, torch.ones_like(masks[name]),torch.zeros_like(masks[name]) )
                idx = torch.multinomial( temp.flatten().to(device),num_remove[name], replacement=False)
                new_masks[name].view(-1)[idx]=1
    return new_masks

def create_mask_regularization_dic(local_masks,cluster_ids,weight_regularization):
    unique_cluster_ids = np.unique(cluster_ids)
    mask_regularization_dic = {}
    n_client = len(local_masks)
    

    for cluster_id in unique_cluster_ids:

        mask_regularization ={}
        for name,tensor in local_masks[0].items():
            mask_regularization[name] = torch.zeros_like(tensor)

            for idx in range(n_client):
                if cluster_ids[idx] == cluster_id:
                    mask_regularization[name] += local_masks[idx][name]
                else:
                    
                    # Reduce sharing network from different clusters
                    mask_regularization[name] -= local_masks[idx][name]
            mask_regularization[name] /= n_client
            mask_regularization[name] *= weight_regularization
        mask_regularization_dic[cluster_id] = mask_regularization

    return mask_regularization_dic

def screen_gradients(model, train_data, device):
    
    model.to(device)
    
    # # # train and update
    criterion = nn.CrossEntropyLoss().to(device)
    # # sample one epoch  of data
    model.zero_grad()
    (x, labels) = next(iter(train_data))
    x, labels = x.to(device), labels.to(device)
    log_probs = model.forward(x)
    loss = criterion(log_probs, labels.long())
    loss.backward()
    gradient={}
    for name, param in model.named_parameters():
        gradient[name] = param.grad.to("cpu")
    return gradient

def apply_mask_dict(mask, model, w_server):  
    model.load_state_dict(w_server)
    
    for name, param in model.named_parameters(): 
        if "weight" in name: 
            weight_dev = param.device
            

            param.data = torch.from_numpy(mask[name].numpy() * w_server[name].cpu().numpy()).to(weight_dev)
            
        if "bias" in name:
            param.data = w_server[name]
    return model.state_dict()

def update_global_model(w_server, w_clients, masks):
    '''
    This function performs Sub-FedAvg-U (for unstructured pruning) as stated in the paper. 
    This function updates the server model based on Sub-FedAvg. It is called at the end of each round. 
    
    :param w_server: server model's state_dict 
    :param w_clients: list of clients' model state_dict to be averaged 
    :param masks: list of clients' pruning masks to be averaged 
    
    :return w_server: updated server model's state_dict
    '''
    
    for name in w_server.keys():
        
        if 'weight' in name:
            
            weight_dev = w_server[name].device
            
            indices = []
            count = np.zeros_like(masks[0][name].reshape([-1]))
            avg = np.zeros_like(w_server[name].data.cpu().numpy().reshape([-1]))
            for i in range(len(masks)): 
                
                count += masks[i][name].numpy().reshape([-1])
                

                avg += w_clients[i][name].data.cpu().numpy().reshape([-1])
            
            final_avg = np.divide(avg, count)         
            ind = np.isfinite(final_avg)
            
            temp_server = w_server[name].data.cpu().numpy().reshape([-1])
            temp_server[ind] = final_avg[ind]
            
            #print(f'Name: {name}, NAN: {np.mean(np.isnan(temp_server))}, INF: {np.mean(np.isinf(temp_server))}')
            
            shape = w_server[name].data.cpu().numpy().shape
            w_server[name].data = torch.from_numpy(temp_server.reshape(shape)).to(weight_dev)            
            
            
        else: 
            
            avg = np.zeros_like(w_server[name].data.cpu().numpy().reshape([-1]))
            for i in range(len(masks)): 
                avg += w_clients[i][name].data.cpu().numpy().reshape([-1])
            avg /= len(masks)
            
            #print(f'Name: {name}, NAN: {np.mean(np.isnan(avg))}, INF: {np.mean(np.isinf(avg))}')
            weight_dev = w_server[name].device
            shape = w_server[name].data.cpu().numpy().shape
            w_server[name].data = torch.from_numpy(avg.reshape(shape)).to(weight_dev)            
            
    return w_server

def mask_hamming_distance(mask_a, mask_b):
    dis = 0; total = 0
    for key in mask_a:
        dis += torch.sum(mask_a[key].int() ^ mask_b[key].int())
        total += mask_a[key].numel()
    return dis, total


def create_data_mask_positive_transfer_correlation(args,clients,initial_state_dict,cluster_ids = [0,0],sparse_value = 0.2,training_rounds = 100,weight_regularization_list=[-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.0000001,-0.0000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],anneal_factor=0.5):
    
    # weight_regularization_list=[-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00001,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.00005,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.0000001,-0.0000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,-0.000001,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    weight_regularization_list= [0,0,0,0,0.01,0.01,0.01,0.01,0.009,0.008,0.007,0.006,0.005,0.001,0.0005,0.0001,0.01,0.01,0.01,0.009,0.008,0.007,0.006,0.005,0.001,0.0005,0.0001,0.01,0.01,0.01,0.009,0.008,0.007,0.006,0.005,0.001,0.0005,0.0001]
    weight_regularization_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # weight_regularization_list = [0,0,0,0,0,0,0,0,0,0,0.001,0.002,0.003,0.001,0.002,0.003,0.001,0.002,0.003,0.001,0.002,0.003]
    # weight_regularization_list = [0,0,0,0,0,0,0,0,0,0,0.0005,0.0005,0.0005,0.0005,0.0005,0.0005,0.0005,0.0005,0.0005,0.0005]

    

    # Initialize sparse mask
    n_client = len(clients)
    mask_distance_list = []
    accuracy_list = []
    counter = 0

    for weight_regularization in weight_regularization_list:
        best_acc = 0
        for i_round in range(training_rounds):
            print("round= ",i_round)
            w_locals = []
            local_masks = []
            masks = []
            # cluster_ids = [0,1]
            
            if i_round >= 1:
                for idx in range(n_client):
                    local_masks.append(copy.deepcopy(clients[idx].get_mask()))
            else:
                for idx in range(n_client):
                    sparsities = calculate_sparsities(initial_state_dict, sparse = sparse_value)
                    mask = init_masks(clients[idx].get_state_dict(),sparsities)
                    dic = apply_mask_dict(mask, 
                                            copy.deepcopy(clients[idx].get_net()), initial_state_dict)
                    clients[idx].set_state_dict(dic) 
                    server_state_dict = copy.deepcopy(initial_state_dict)
                    local_masks.append(mask)
                    clients[idx].set_mask(mask)

            mask_regularization_dic = create_mask_regularization_dic(local_masks,cluster_ids,weight_regularization)


            for idx in range(n_client):
            # Update local models based on a global model
                if i_round >= 1:
                    dic = apply_mask_dict(copy.deepcopy(clients[idx].get_mask()), 
                                            copy.deepcopy(clients[idx].get_net()), server_state_dict)
                    
                    clients[idx].set_state_dict(dic) 
                            

            # Pruning with the regularization term
                clients[idx].local_train()
                new_masks, num_remove = fire_mask(copy.deepcopy(clients[idx].get_mask()), copy.deepcopy(clients[idx].get_state_dict()),None ,i_round,training_rounds/2,mask_regularization_dic[cluster_ids[idx]],anneal_factor)
                dic = apply_mask_dict(new_masks, 
                                            copy.deepcopy(clients[idx].get_net()), copy.deepcopy(clients[idx].get_state_dict()))
                clients[idx].set_state_dict(dic) 

            # Grow with the regularization term
                gradients = screen_gradients(copy.deepcopy(clients[idx].get_net()), clients[idx].get_training_data(), None)
                new_masks = regrow_mask(new_masks,  num_remove,None ,mask_regularization_dic[cluster_ids[idx]],gradients,dis_gradient_check=None)
                dic = apply_mask_dict(new_masks, 
                                            copy.deepcopy(clients[idx].get_net()), copy.deepcopy(clients[idx].get_state_dict()))
                clients[idx].set_state_dict(dic) 
                clients[idx].set_mask(new_masks)
                masks.append(copy.deepcopy(clients[idx].get_mask()))     
                w_locals.append(copy.deepcopy(clients[idx].get_state_dict()))

            # Update a global model

            server_state_dict = update_global_model(server_state_dict, w_locals, masks)

            # Save the best test accuracy
            acc_sum = 0 
            for idx in range(n_client):
                loss,acc = clients[idx].eval_test()
                acc_sum += acc
            acc_avg = acc_sum / float(n_client)
            print("acc_avg",acc_avg)
            if best_acc < acc_avg:
                best_acc = acc_avg
                



        # save data
        mask_distance_list.append([])

        final_mask_a = clients[0].get_mask()
        final_mask_b = clients[1].get_mask()

        total_dis,total_num = mask_hamming_distance(final_mask_a,final_mask_b)
        mask_distance_list[counter].append((total_dis/total_num).detach().numpy().item())

        mask_a_id = 0
        
        for mask_a_layer_wise in final_mask_a.items():

            mask_b_id = 0
            for mask_b_layer_wise in final_mask_b.items():
                if mask_a_id == mask_b_id:
                    

                    total_dis,total_num = mask_hamming_distance({mask_a_layer_wise[0]:mask_a_layer_wise[1]},{mask_b_layer_wise[0]:mask_b_layer_wise[1]})
                    mask_distance_list[counter].append((total_dis/total_num).detach().numpy().item())
                    


                mask_b_id += 1
            mask_a_id += 1

        param_a = clients[0].get_state_dict()
        param_b = clients[1].get_state_dict()

        for tensor in param_a.items():      
            weight_1 = torch.flatten(tensor[1])
            weight_2 = torch.flatten(param_b[tensor[0]])
        
            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            
            cosine_sim = cos(weight_1, weight_2)
            
            mask_distance_list[counter].append(cosine_sim.detach().numpy().item())

        counter_tensor = 0
        for tensor in param_a.items():
            if "weight" in tensor[0]:
                if counter_tensor == 0:
                    weight_1_all = torch.flatten(tensor[1])
                    counter_tensor += 1
                else:
                    weight_1_all = torch.cat((weight_1, torch.flatten(tensor[1])), 0)
        counter_tensor = 0
        for tensor in param_b.items():
            if "weight" in tensor[0]:
                if counter_tensor == 0:
                    weight_2_all = torch.flatten(tensor[1])
                    counter_tensor += 1
                else:
                    weight_2_all = torch.cat((weight_2, torch.flatten(tensor[1])), 0)

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        cosine_sim = cos(weight_1_all, weight_2_all)
        

        mask_distance_list[counter].append(cosine_sim.detach().numpy().item())
        
        

        accuracy_list.append(best_acc.detach().numpy().item())



        counter += 1
        

    file_name_common = "src/data/positive_transfer/dataset_" + str(args.dataset) + "training_round_" + str(training_rounds) + "anneal_factor_" +str(anneal_factor) + "similar_clients_" + str(args.transfer) + "sparcities_" + str(sparse_value) +"algorithm_" + str(args.algorithm)

    sim_file_name = file_name_common + "mask_distance.csv"
    with open(sim_file_name, 'w') as f:
    # using csv.writer method from CSV package
        write = csv.writer(f)
        
        write.writerows(mask_distance_list)

    pt_file_name = file_name_common + "accuracy_list.csv"
    with open(pt_file_name, 'w') as f:
    # using csv.writer method from CSV package
        write = csv.writer(f)
        
        write.writerows([accuracy_list])
        

